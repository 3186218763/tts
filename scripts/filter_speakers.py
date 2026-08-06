"""Build a robust Huayin voice reference and score acoustic-filtered slices.

The reference is inferred from the speaker that recurs across the beginning of
all whitelisted chat recordings.  Results are three-state: only ``accept`` is
used for the final dataset, while ``uncertain`` remains available for later
review or ASR backfilling.  ``reject`` is enabled only when pseudo-negative and
cross-source positive score distributions have a clear gap.

Embeddings default to CUDA (one worker process per GPU). CPU multi-process
remains available as a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ERES2NET_DIR = Path("/home/mtr/tt/GPT-SoVITS/GPT_SoVITS/eres2net")
MODEL_PATH = Path(
    "/home/mtr/tt/GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/"
    "pretrained_eres2netv2w24s4ep4.ckpt"
)
EMBEDDING_CACHE = PROJECT_ROOT / "data" / "speaker_embeddings.npz"
CALIBRATION_PATH = PROJECT_ROOT / "data" / "speaker_calibration.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "speaker_results.json"

_MODEL: Any = None
_KALDI: Any = None
_TORCHAUDIO: Any = None
_SOUNDFILE: Any = None
_TORCH: Any = None
_DEVICE: Any = None


def _import_pipeline():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import build_dataset

    return build_dataset


def _source_name(path: str | Path) -> str:
    return Path(path).name.partition("_(Vocals)")[0]


def _slice_start(path: str | Path) -> int:
    stem = Path(path).stem
    try:
        return int(stem.rsplit("__", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return 0


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _model_sha256() -> str:
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _init_embedding_worker(
    threads: int, device: str = "cpu", gpu_id: str | None = None
) -> None:
    global _MODEL, _KALDI, _TORCHAUDIO, _SOUNDFILE, _TORCH, _DEVICE

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    if device == "cuda" and gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif device != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import soundfile
    import torchaudio

    if str(ERES2NET_DIR) not in sys.path:
        sys.path.insert(0, str(ERES2NET_DIR))
    from ERes2NetV2 import ERes2NetV2
    import kaldi

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        # After CUDA_VISIBLE_DEVICES remapping, the selected GPU is always :0.
        torch_device = torch.device("cuda:0")
    else:
        torch_device = torch.device("cpu")

    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    model.load_state_dict(state)
    model.to(torch_device)
    model.eval()
    _MODEL = model
    _KALDI = kaldi
    _TORCHAUDIO = torchaudio
    _SOUNDFILE = soundfile
    _TORCH = torch
    _DEVICE = torch_device


def _embed_one(path_value: str) -> tuple[str, np.ndarray]:
    path = Path(path_value)
    samples, sample_rate = _SOUNDFILE.read(
        str(path), dtype="float32", always_2d=True
    )
    waveform = _TORCH.from_numpy(samples.T).float().mean(dim=0, keepdim=True)
    if sample_rate != 16_000:
        waveform = _TORCHAUDIO.functional.resample(
            waveform, sample_rate, 16_000
        )
    # Kaldi fbank stays on CPU; only the network runs on GPU.
    features = _KALDI.fbank(
        waveform,
        num_mel_bins=80,
        sample_frequency=16_000,
        dither=0,
    )
    with _TORCH.inference_mode():
        embedding = _MODEL(features.unsqueeze(0).to(_DEVICE))[0]
    vector = embedding.detach().float().cpu().numpy().astype(np.float32)
    return str(path), _normalize(vector)


def _load_embedding_cache() -> dict[str, np.ndarray]:
    if not EMBEDDING_CACHE.exists():
        return {}
    with np.load(EMBEDDING_CACHE, allow_pickle=False) as data:
        paths = data["paths"].astype(str).tolist()
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(paths):
        raise ValueError(f"invalid embedding cache: {EMBEDDING_CACHE}")
    return {path: _normalize(vector) for path, vector in zip(paths, embeddings)}


def _write_embedding_cache(cache: dict[str, np.ndarray]) -> None:
    EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(cache)
    embeddings = np.stack([cache[path] for path in paths]).astype(np.float32)
    temporary = EMBEDDING_CACHE.with_name(f"{EMBEDDING_CACHE.stem}.tmp.npz")
    np.savez(temporary, paths=np.asarray(paths), embeddings=embeddings)
    temporary.replace(EMBEDDING_CACHE)


def _gpu_worker(
    worker_label: str,
    gpu_id: str,
    threads: int,
    paths: list[str],
    result_queue: Any,
) -> None:
    try:
        _init_embedding_worker(threads, device="cuda", gpu_id=gpu_id)
        for path_value in paths:
            result_queue.put(("result", worker_label, _embed_one(path_value)))
    except BaseException:
        result_queue.put(("error", worker_label, traceback.format_exc()))
    finally:
        result_queue.put(("done", worker_label, None))


def _extract_embeddings_cuda(
    pending: list[Path],
    cache: dict[str, np.ndarray],
    *,
    gpus: list[str],
    threads_per_worker: int,
) -> None:
    if not gpus:
        raise ValueError("at least one GPU id is required for cuda embeddings")
    shards = [pending[index :: len(gpus)] for index in range(len(gpus))]
    assignments = [
        (f"GPU {gpu_id}", gpu_id, shard)
        for gpu_id, shard in zip(gpus, shards)
        if shard
    ]
    print(
        f"speaker embeddings: {len(pending)} pending on CUDA "
        f"({len(assignments)} GPU workers)",
        flush=True,
    )
    for label, gpu_id, shard in assignments:
        print(f"  {label}: {len(shard)} slices", flush=True)

    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    workers = [
        context.Process(
            target=_gpu_worker,
            args=(
                label,
                gpu_id,
                threads_per_worker,
                [str(path) for path in shard],
                result_queue,
            ),
            name=f"speaker-{label.lower().replace(' ', '-')}",
        )
        for label, gpu_id, shard in assignments
    ]
    for worker in workers:
        worker.start()

    completed = 0
    completed_workers = 0
    errors: list[str] = []
    try:
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
                path_value, embedding = payload
                cache[path_value] = embedding
                completed += 1
                if completed % 50 == 0 or completed == len(pending):
                    _write_embedding_cache(cache)
                    print(
                        f"  embedded {completed}/{len(pending)} ({worker_label})",
                        flush=True,
                    )
            elif kind == "error":
                errors.append(f"{worker_label}:\n{payload}")
            elif kind == "done":
                completed_workers += 1
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        if cache:
            _write_embedding_cache(cache)
    if errors:
        raise RuntimeError("\n".join(errors))


def _extract_embeddings_cpu(
    pending: list[Path],
    cache: dict[str, np.ndarray],
    *,
    workers: int,
    threads_per_worker: int,
) -> None:
    print(
        f"speaker embeddings: {len(pending)} pending, "
        f"{workers} workers x {threads_per_worker} CPU threads",
        flush=True,
    )
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_embedding_worker,
        initargs=(threads_per_worker, "cpu", None),
    ) as executor:
        futures = {
            executor.submit(_embed_one, str(path)): path for path in pending
        }
        for future in as_completed(futures):
            path_value, embedding = future.result()
            cache[path_value] = embedding
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                _write_embedding_cache(cache)
                print(
                    f"  embedded {completed}/{len(pending)}",
                    flush=True,
                )


def extract_embeddings(
    paths: list[Path],
    *,
    workers: int,
    threads_per_worker: int,
    device: str = "auto",
    gpus: list[str] | None = None,
) -> dict[str, np.ndarray]:
    cache = _load_embedding_cache()
    pending = [path for path in paths if str(path) not in cache]
    if not pending:
        print(f"speaker embeddings: all {len(paths)} cached", flush=True)
        return {str(path): cache[str(path)] for path in paths}

    runtime_device = device
    if runtime_device == "auto":
        import torch

        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if runtime_device == "cuda":
            _extract_embeddings_cuda(
                pending,
                cache,
                gpus=list(gpus or ["0"]),
                threads_per_worker=threads_per_worker,
            )
        else:
            _extract_embeddings_cpu(
                pending,
                cache,
                workers=min(workers, len(pending)),
                threads_per_worker=threads_per_worker,
            )
    except BaseException:
        if cache:
            _write_embedding_cache(cache)
        raise
    _write_embedding_cache(cache)
    return {str(path): cache[str(path)] for path in paths}


def _reference_candidates(
    records: list[dict[str, Any]], per_source: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        duration = float(record.get("duration") or 0.0)
        rms_db = float(record.get("rms_db") or -120.0)
        if 2.5 <= duration <= 12.0 and rms_db >= -35.0:
            grouped[_source_name(record["path"])].append(record)
    for source, items in grouped.items():
        items.sort(key=lambda item: _slice_start(item["path"]))
        grouped[source] = items[:per_source]
    return dict(grouped)


def _cross_source_scores(
    vectors: np.ndarray, sources: list[str]
) -> np.ndarray:
    similarities = vectors @ vectors.T
    unique_sources = sorted(set(sources))
    scores = np.empty(len(vectors), dtype=np.float32)
    for index, source in enumerate(sources):
        per_source = []
        for other_source in unique_sources:
            if other_source == source:
                continue
            candidates = [
                other_index
                for other_index, value in enumerate(sources)
                if value == other_source
            ]
            per_source.append(float(np.max(similarities[index, candidates])))
        scores[index] = float(np.median(per_source))
    return scores


def calibrate_reference(
    candidate_groups: dict[str, list[dict[str, Any]]],
    embeddings: dict[str, np.ndarray],
    *,
    references_per_source: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    candidates = [
        record
        for source in sorted(candidate_groups)
        for record in candidate_groups[source]
    ]
    sources = [_source_name(record["path"]) for record in candidates]
    if len(set(sources)) < 3:
        raise SystemExit("speaker calibration needs at least three source recordings")
    vectors = np.stack([embeddings[str(record["path"])] for record in candidates])
    cross_scores = _cross_source_scores(vectors, sources)

    initial_indices: list[int] = []
    for source in sorted(set(sources)):
        indices = [index for index, value in enumerate(sources) if value == source]
        indices.sort(key=lambda index: float(cross_scores[index]), reverse=True)
        initial_indices.extend(indices[: max(references_per_source * 2, 8)])

    initial_centers: dict[str, np.ndarray] = {}
    for source in sorted(set(sources)):
        selected = [index for index in initial_indices if sources[index] == source]
        initial_centers[source] = _normalize(np.mean(vectors[selected], axis=0))

    refined_scores = np.empty(len(candidates), dtype=np.float32)
    for index, source in enumerate(sources):
        values = [
            float(vectors[index] @ center)
            for other_source, center in initial_centers.items()
            if other_source != source
        ]
        refined_scores[index] = float(np.median(values))

    reference_indices: list[int] = []
    for source in sorted(set(sources)):
        indices = [index for index, value in enumerate(sources) if value == source]
        indices.sort(key=lambda index: float(refined_scores[index]), reverse=True)
        reference_indices.extend(indices[:references_per_source])

    source_centers: dict[str, np.ndarray] = {}
    for source in sorted(set(sources)):
        selected = [index for index in reference_indices if sources[index] == source]
        source_centers[source] = _normalize(np.mean(vectors[selected], axis=0))

    positive_scores = []
    for index in reference_indices:
        source = sources[index]
        other_centers = [
            center
            for other_source, center in source_centers.items()
            if other_source != source
        ]
        leave_one_source_out = _normalize(np.mean(other_centers, axis=0))
        positive_scores.append(float(vectors[index] @ leave_one_source_out))
    accept_threshold = float(np.percentile(positive_scores, 5) - 0.02)

    global_center = _normalize(np.mean(list(source_centers.values()), axis=0))
    non_reference = [
        index for index in range(len(candidates)) if index not in set(reference_indices)
    ]
    low_cross = sorted(non_reference, key=lambda index: float(cross_scores[index]))
    low_cross = low_cross[: max(8, len(low_cross) // 5)]
    negative_scores = [float(vectors[index] @ global_center) for index in low_cross]
    reject_threshold = (
        float(np.percentile(negative_scores, 99)) if negative_scores else -1.0
    )
    reject_enabled = reject_threshold + 0.03 < accept_threshold

    calibration = {
        "version": 1,
        "model_path": str(MODEL_PATH),
        "model_sha256": _model_sha256(),
        "sources": sorted(source_centers),
        "candidate_count": len(candidates),
        "reference_count": len(reference_indices),
        "references": [str(candidates[index]["path"]) for index in reference_indices],
        "accept_threshold": accept_threshold,
        "reject_threshold": reject_threshold,
        "reject_enabled": reject_enabled,
        "positive_score_percentiles": {
            str(percentile): float(np.percentile(positive_scores, percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
        "pseudo_negative_score_percentiles": {
            str(percentile): float(np.percentile(negative_scores, percentile))
            for percentile in (1, 5, 50, 95, 99)
        }
        if negative_scores
        else {},
    }
    return calibration, source_centers


def score_slices(
    records: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    calibration: dict[str, Any],
    source_centers: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260803)
    centers = np.stack([source_centers[source] for source in sorted(source_centers)])
    bootstrap_centers = []
    for _ in range(128):
        sample = centers[rng.integers(0, len(centers), size=len(centers))]
        bootstrap_centers.append(_normalize(np.mean(sample, axis=0)))
    bootstrap_matrix = np.stack(bootstrap_centers)

    accept_threshold = float(calibration["accept_threshold"])
    reject_threshold = float(calibration["reject_threshold"])
    reject_enabled = bool(calibration["reject_enabled"])
    results = []
    for record in records:
        path_value = str(record["path"])
        vector = embeddings[path_value]
        per_source = centers @ vector
        bootstrap_scores = bootstrap_matrix @ vector
        score = float(np.median(per_source))
        lower = float(np.percentile(bootstrap_scores, 5))
        upper = float(np.percentile(bootstrap_scores, 95))
        if lower >= accept_threshold:
            decision = "accept"
        elif reject_enabled and upper <= reject_threshold:
            decision = "reject"
        else:
            decision = "uncertain"
        results.append(
            {
                "path": path_value,
                "decision": decision,
                "score": score,
                "score_lower": lower,
                "score_upper": upper,
                "source_scores": {
                    source: float(value)
                    for source, value in zip(sorted(source_centers), per_source)
                },
            }
        )
    return results


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="embedding device (default: auto → CUDA if available)",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0", "1"],
        help="CUDA device IDs, one embedding worker per ID (default: 0 1)",
    )
    parser.add_argument("--reference-candidates-per-source", type=int, default=40)
    parser.add_argument("--references-per-source", type=int, default=8)
    args = parser.parse_args()
    if min(
        args.workers,
        args.threads_per_worker,
        args.reference_candidates_per_source,
        args.references_per_source,
    ) < 1:
        parser.error("all numeric options must be positive")
    if not args.gpus:
        parser.error("at least one GPU ID is required")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be unique")

    pipeline = _import_pipeline()
    filter_records = pipeline._read_json_list(
        pipeline.FILTER_RESULTS, required=True
    )
    keep_records = [
        record
        for record in filter_records
        if record.get("path") and record.get("keep") is True
    ]
    selected_slices = pipeline._select_training_derivatives(
        sorted(pipeline.SLICES_DIR.glob("*.wav"))
    )
    keep_by_path = {
        pipeline.canonical_path(record["path"]): record for record in keep_records
    }
    paths = [path for path in selected_slices if pipeline.canonical_path(path) in keep_by_path]
    if not paths:
        raise SystemExit("no acoustic-filtered training slices")
    ordered_records = [keep_by_path[pipeline.canonical_path(path)] for path in paths]

    embeddings = extract_embeddings(
        paths,
        workers=min(args.workers, len(paths)),
        threads_per_worker=args.threads_per_worker,
        device=args.device,
        gpus=args.gpus,
    )
    candidate_groups = _reference_candidates(
        ordered_records, args.reference_candidates_per_source
    )
    calibration, source_centers = calibrate_reference(
        candidate_groups,
        embeddings,
        references_per_source=args.references_per_source,
    )
    results = score_slices(
        ordered_records, embeddings, calibration, source_centers
    )
    _write_json(CALIBRATION_PATH, calibration)
    _write_json(RESULTS_PATH, results)

    decisions: dict[str, int] = defaultdict(int)
    for result in results:
        decisions[str(result["decision"])] += 1
    print(
        "speaker calibration: "
        f"accept>={calibration['accept_threshold']:.4f}, "
        f"reject<={calibration['reject_threshold']:.4f}, "
        f"reject_enabled={calibration['reject_enabled']}",
        flush=True,
    )
    print(f"speaker results: {dict(sorted(decisions.items()))}", flush=True)
    print(f"saved: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
