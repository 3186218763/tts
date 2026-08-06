"""Run filter-aware faster-whisper transcription across GPUs or CPU workers.

The parent process owns ``data/asr_results.json`` and writes every completed
record atomically. Workers only transcribe their disjoint slice shards,
so an interrupted run can continue from the existing cache.
"""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import queue
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ASR_LOCK = PROJECT_ROOT / "data" / "asr_results.lock"


@contextmanager
def _exclusive_asr_run():
    """Serialize ASR parents so their atomic cache writes cannot race."""
    ASR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with ASR_LOCK.open("a+", encoding="utf-8") as lock_file:
        print(f"waiting for ASR lock: {ASR_LOCK}", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print("ASR lock acquired", flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _import_pipeline():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import build_dataset

    return build_dataset


def _worker(
    worker_label: str,
    device: str,
    gpu_id: str | None,
    cpu_threads: int,
    beam_size: int,
    model_size: str,
    paths: list[str],
    result_queue: Any,
) -> None:
    try:
        os.environ["HUAYIN_ASR_DEVICE"] = device
        if device == "cuda" and gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["HUAYIN_ASR_CPU_THREADS"] = str(cpu_threads)
        pipeline = _import_pipeline()
        model = pipeline._load_whisper_model(model_size)
        for path_value in paths:
            path = Path(path_value)
            segments, info = model.transcribe(
                str(path),
                beam_size=beam_size,
                best_of=1 if beam_size == 1 else 5,
                temperature=0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                language=None,
            )
            result_queue.put(
                (
                    "result",
                    worker_label,
                    pipeline.make_asr_record(path, segments, info),
                )
            )
    except BaseException:
        result_queue.put(("error", worker_label, traceback.format_exc()))
    finally:
        result_queue.put(("done", worker_label, None))


def _eligible_state(pipeline, speaker_policy: str):
    all_slices = sorted(pipeline.SLICES_DIR.glob("*.wav"))
    if not all_slices:
        raise SystemExit(
            f"no slices under {pipeline.SLICES_DIR}; run the slice stage first"
        )
    slices = pipeline._select_training_derivatives(all_slices)
    filter_results = pipeline._read_json_list(
        pipeline.FILTER_RESULTS, required=True
    )
    eligible = pipeline.select_asr_slices(
        slices, filter_results, speaker_policy=speaker_policy
    )
    cached = pipeline._read_json_list(pipeline.ASR_RESULTS)
    by_path: dict[Path, dict[str, Any]] = {}
    ordered_keys: list[Path] = []
    for item in cached:
        if not item.get("path"):
            continue
        key = pipeline.canonical_path(item["path"])
        if key not in by_path:
            by_path[key] = item
            ordered_keys.append(key)
    pending = [
        path
        for path in eligible
        if not pipeline.has_asr_quality(
            by_path.get(pipeline.canonical_path(path), {})
        )
    ]
    return eligible, by_path, pending, ordered_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0", "1"],
        help="CUDA device IDs, one ASR worker per ID (default: 0 1)",
    )
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="execution device (default: auto)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="CPU worker processes when --device=cpu (default: up to 4)",
    )
    parser.add_argument("--cpu-threads-per-worker", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--speaker-policy",
        choices=["accept", "accept-or-uncertain"],
        default="accept-or-uncertain",
    )
    args = parser.parse_args()

    if not args.gpus:
        parser.error("at least one GPU ID is required")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be unique")
    if min(args.workers, args.cpu_threads_per_worker, args.beam_size) < 1:
        parser.error("CPU worker and thread counts must be positive")

    with _exclusive_asr_run():
        _run(args)


def _run(args: argparse.Namespace) -> None:
    pipeline = _import_pipeline()
    eligible, by_path, pending, ordered_keys = _eligible_state(
        pipeline, args.speaker_policy
    )

    def ordered_results() -> list[dict[str, Any]]:
        return [by_path[key] for key in ordered_keys if key in by_path]

    if not pending:
        pipeline._write_json(pipeline.ASR_RESULTS, ordered_results())
        print(f"no new slices; cached {len(by_path)} records")
        return

    runtime_device = args.device
    if runtime_device == "auto":
        import torch

        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    worker_specs = (
        [(f"GPU {gpu_id}", gpu_id) for gpu_id in args.gpus]
        if runtime_device == "cuda"
        else [
            (f"CPU {index}", None)
            for index in range(min(args.workers, len(pending)))
        ]
    )
    shards = [
        pending[index :: len(worker_specs)] for index in range(len(worker_specs))
    ]
    assignments = [
        (worker_label, gpu_id, shard)
        for (worker_label, gpu_id), shard in zip(worker_specs, shards, strict=True)
        if shard
    ]
    print(
        f"transcribing {len(pending)} slices with {len(assignments)} "
        f"{runtime_device.upper()} workers",
        flush=True,
    )
    for worker_label, _, shard in assignments:
        print(f"  {worker_label}: {len(shard)} slices", flush=True)

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_worker,
            args=(
                worker_label,
                runtime_device,
                gpu_id,
                args.cpu_threads_per_worker,
                args.beam_size,
                args.whisper_model,
                [str(path) for path in shard],
                result_queue,
            ),
            name=f"asr-{worker_label.lower().replace(' ', '-')}",
        )
        for worker_label, gpu_id, shard in assignments
    ]
    for worker in workers:
        worker.start()

    completed_workers = 0
    completed_slices = 0
    errors: list[str] = []
    while completed_workers < len(workers):
        try:
            kind, worker_label, payload = result_queue.get(timeout=5)
        except queue.Empty:
            crashed = [
                worker
                for worker in workers
                if worker.exitcode not in (None, 0)
            ]
            if crashed:
                errors.extend(
                    f"{worker.name} exited with {worker.exitcode}"
                    for worker in crashed
                )
                break
            continue

        if kind == "result":
            item = payload
            key = pipeline.canonical_path(item["path"])
            if key not in by_path:
                ordered_keys.append(key)
            by_path[key] = item
            completed_slices += 1
            if completed_slices % 25 == 0:
                pipeline._write_json(pipeline.ASR_RESULTS, ordered_results())
            print(
                f"  [{completed_slices}/{len(pending)}] {worker_label} "
                f"{Path(item['path']).name} ({item['lang']}) {item['text'][:50]}",
                flush=True,
            )
        elif kind == "error":
            errors.append(f"{worker_label}:\n{payload}")
        elif kind == "done":
            completed_workers += 1

    for worker in workers:
        worker.join(timeout=10)
    if errors:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        pipeline._write_json(pipeline.ASR_RESULTS, ordered_results())
        raise SystemExit("\n".join(errors))

    pipeline._write_json(pipeline.ASR_RESULTS, ordered_results())
    print(f"ASR complete: {len(by_path)} cached records")


if __name__ == "__main__":
    main()
