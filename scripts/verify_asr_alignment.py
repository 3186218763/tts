"""Forced-language re-ASR to verify speech↔text consistency.

For each speaker-accepted JP/ZH transcript, re-run faster-whisper with the
claimed language forced, then compare the forced transcript against the
cached free-language text.  Large divergence is a strong hallucination signal.

Results are written to ``data/alignment_results.json`` (incremental) and can
be consumed by ``build_dataset.build_dataset(require_alignment=True)``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ALIGNMENT_PATH = PROJECT_ROOT / "data" / "alignment_results.json"
ALIGNMENT_VERSION = 1


def _import_pipeline():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import build_dataset
    import dataset_text_quality as text_quality

    return build_dataset, text_quality


def _atomic_write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _forced_lang(lang: str) -> str | None:
    pipeline, text_quality = _import_pipeline()
    normalized = text_quality.normalize_lang(lang)
    if normalized in {"ja", "zh", "en"}:
        return normalized
    return None


def _candidate_items(
    pipeline,
    text_quality,
    *,
    allow_en: bool,
) -> list[dict[str, Any]]:
    transcripts = pipeline._read_json_list(pipeline.ASR_RESULTS, required=True)
    filters = {
        pipeline.canonical_path(item["path"]): item.get("keep") is True
        for item in pipeline._read_json_list(pipeline.FILTER_RESULTS, required=True)
        if item.get("path")
    }
    speakers = {}
    if pipeline.SPEAKER_RESULTS.exists():
        speakers = {
            pipeline.canonical_path(item["path"]): str(item.get("decision") or "")
            for item in pipeline._read_json_list(
                pipeline.SPEAKER_RESULTS, required=True
            )
            if item.get("path")
        }

    candidates: list[dict[str, Any]] = []
    for item in transcripts:
        if not item.get("path"):
            continue
        src = pipeline.canonical_path(item["path"])
        if filters.get(src) is not True:
            continue
        if speakers and speakers.get(src) != "accept":
            continue
        lang = text_quality.normalize_lang(item.get("lang"))
        if lang == "en" and not allow_en:
            continue
        if lang not in {"ja", "zh", "en"}:
            continue
        text = str(item.get("text") or "").strip()
        gate = text_quality.evaluate_text_quality(text, lang, allow_en=allow_en)
        if not gate.keep:
            continue
        # Mirror build_dataset ASR quality gates so we don't burn GPU on
        # samples that would be discarded before alignment is consulted.
        if pipeline.has_asr_quality(item):
            audio_duration = pipeline._finite_float(item.get("audio_duration"))
            if audio_duration and audio_duration > 0:
                nonspace_length = len("".join(text.split()))
                if nonspace_length / audio_duration > 12.0:
                    continue
            language_probability = pipeline._finite_float(
                item.get("language_probability")
            )
            avg_logprob = pipeline._finite_float(item.get("avg_logprob"))
            no_speech_probability = pipeline._finite_float(
                item.get("max_no_speech_probability")
            )
            compression_ratio = pipeline._finite_float(
                item.get("max_compression_ratio")
            )
            if (
                language_probability is not None
                and language_probability < pipeline.MIN_LANGUAGE_PROBABILITY
            ):
                continue
            if avg_logprob is None or avg_logprob < pipeline.MIN_AVG_LOGPROB:
                continue
            if (
                no_speech_probability is not None
                and no_speech_probability > pipeline.MAX_NO_SPEECH_PROBABILITY
            ):
                continue
            if (
                compression_ratio is not None
                and compression_ratio > pipeline.MAX_COMPRESSION_RATIO
            ):
                continue
        if not src.is_file() or src.stat().st_size < 8_000:
            continue
        candidates.append(
            {
                "path": str(src),
                "lang": lang,
                "text": text,
            }
        )
    return candidates


def _worker(
    worker_label: str,
    gpu_id: str,
    model_size: str,
    beam_size: int,
    min_similarity: float,
    jobs: list[dict[str, Any]],
    result_queue: Any,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["HUAYIN_ASR_DEVICE"] = "cuda"
        pipeline, text_quality = _import_pipeline()
        model = pipeline._load_whisper_model(model_size)
        for job in jobs:
            path = Path(job["path"])
            lang = job["lang"]
            original = job["text"]
            try:
                segments, info = model.transcribe(
                    str(path),
                    beam_size=beam_size,
                    best_of=1 if beam_size == 1 else 5,
                    temperature=0,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                    language=lang,
                )
                record = pipeline.make_asr_record(path, segments, info)
                forced_text = str(record.get("text") or "").strip()
                sim = text_quality.similarity_ratio(original, forced_text)
                forced_gate = text_quality.evaluate_text_quality(
                    forced_text, lang, allow_en=True
                )
                avg_logprob = record.get("avg_logprob")
                no_speech = record.get("max_no_speech_probability")
                reason = None
                keep = True
                if not forced_text:
                    keep = False
                    reason = "empty_forced"
                elif not forced_gate.keep:
                    keep = False
                    reason = f"forced_text_{forced_gate.reasons[0]}"
                elif sim < min_similarity:
                    keep = False
                    reason = "low_similarity"
                elif (
                    avg_logprob is not None
                    and float(avg_logprob) < pipeline.MIN_AVG_LOGPROB
                ):
                    keep = False
                    reason = "forced_logprob"
                elif (
                    no_speech is not None
                    and float(no_speech) > pipeline.MAX_NO_SPEECH_PROBABILITY
                ):
                    keep = False
                    reason = "forced_no_speech"

                result_queue.put(
                    (
                        "result",
                        worker_label,
                        {
                            "path": str(path.resolve()),
                            "lang": lang,
                            "original_text": original,
                            "forced_text": forced_text,
                            "similarity": round(sim, 4),
                            "keep": keep,
                            "reason": reason,
                            "avg_logprob": avg_logprob,
                            "max_no_speech_probability": no_speech,
                            "language_probability": record.get(
                                "language_probability"
                            ),
                            "alignment_version": ALIGNMENT_VERSION,
                        },
                    )
                )
            except BaseException:
                result_queue.put(
                    (
                        "result",
                        worker_label,
                        {
                            "path": str(path.resolve()),
                            "lang": lang,
                            "original_text": original,
                            "forced_text": "",
                            "similarity": 0.0,
                            "keep": False,
                            "reason": "error",
                            "error": traceback.format_exc()[-500:],
                            "alignment_version": ALIGNMENT_VERSION,
                        },
                    )
                )
    except BaseException:
        result_queue.put(("error", worker_label, traceback.format_exc()))
    finally:
        result_queue.put(("done", worker_label, None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument(
        "--allow-en",
        action="store_true",
        help="also verify English-labelled clips (default: JP/ZH only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional cap for smoke tests (0 = all candidates)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-verify paths already present in alignment_results.json",
    )
    args = parser.parse_args()

    pipeline, text_quality = _import_pipeline()
    candidates = _candidate_items(
        pipeline, text_quality, allow_en=args.allow_en
    )
    existing: dict[str, dict[str, Any]] = {}
    if ALIGNMENT_PATH.exists() and not args.force:
        for item in pipeline._read_json_list(ALIGNMENT_PATH, required=False):
            if item.get("path"):
                existing[str(pipeline.canonical_path(item["path"]))] = item

    pending = [
        item
        for item in candidates
        if args.force
        or str(pipeline.canonical_path(item["path"])) not in existing
    ]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"alignment candidates: {len(candidates)}, "
        f"cached: {len(existing)}, pending: {len(pending)}",
        flush=True,
    )
    if not pending:
        print("nothing to verify", flush=True)
        return

    # Balance by count across GPUs.
    shards: list[list[dict[str, Any]]] = [[] for _ in args.gpus]
    for index, item in enumerate(pending):
        shards[index % len(args.gpus)].append(item)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    workers = []
    for gpu_id, shard in zip(args.gpus, shards):
        if not shard:
            continue
        proc = ctx.Process(
            target=_worker,
            args=(
                f"GPU{gpu_id}",
                gpu_id,
                args.whisper_model,
                args.beam_size,
                args.min_similarity,
                shard,
                result_queue,
            ),
        )
        proc.start()
        workers.append(proc)

    done_workers = 0
    total_workers = len(workers)
    completed = 0
    merged = dict(existing)
    while done_workers < total_workers:
        kind, label, payload = result_queue.get()
        if kind == "done":
            done_workers += 1
            print(f"{label} done", flush=True)
            continue
        if kind == "error":
            for proc in workers:
                proc.terminate()
            raise SystemExit(f"worker {label} failed:\n{payload}")
        path_key = str(pipeline.canonical_path(payload["path"]))
        merged[path_key] = payload
        completed += 1
        status = "keep" if payload.get("keep") else f"drop:{payload.get('reason')}"
        print(
            f"  [{completed}/{len(pending)}] {label} "
            f"{Path(payload['path']).name} "
            f"sim={payload.get('similarity')} {status} "
            f"{(payload.get('forced_text') or '')[:40]}",
            flush=True,
        )
        if completed % 50 == 0 or completed == len(pending):
            _atomic_write_json(ALIGNMENT_PATH, list(merged.values()))

    for proc in workers:
        proc.join()

    _atomic_write_json(ALIGNMENT_PATH, list(merged.values()))
    kept = sum(1 for item in merged.values() if item.get("keep"))
    dropped = len(merged) - kept
    print(
        f"alignment complete: {len(merged)} records "
        f"(keep={kept}, drop={dropped}) → {ALIGNMENT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
