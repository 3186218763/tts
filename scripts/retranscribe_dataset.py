#!/usr/bin/env python3
"""Retranscribe curated clips with forced language + Chinese/Japanese prompts.

Fixes the earlier ASR pass (language=None auto-detect, no initial_prompt)
which produced hallucinated/truncated transcripts on clean audio.

Run with the gptsovits env:
  /home/mtr/miniconda3/envs/gptsovits/bin/python scripts/retranscribe_dataset.py --limit 150

Writes data/dataset_clean/asr_retranscribed.json (incremental cache).
Original files are never modified.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

AUDIO_DIR = PROJECT_ROOT / "data" / "dataset_clean" / "audio"
OUT_JSON = PROJECT_ROOT / "data" / "dataset_clean" / "asr_retranscribed.json"

PROMPTS = {
    "zh": "以下是普通话的句子。今天天气真不错，我们一起去公园散步吧。",
    "ja": "以下は日本語の文章です。",
}


def lang_of(path: Path) -> str:
    return "ja" if str(path).endswith("_jp.wav") else "zh"


def load_model(model_size: str, device: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type="float16")


def transcribe_one(model, path: Path) -> dict:
    lang = lang_of(path)
    segments, info = model.transcribe(
        str(path),
        language=lang,
        initial_prompt=PROMPTS[lang],
        beam_size=5,
        best_of=5,
        temperature=0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    materialized = list(segments)
    text = "".join(str(getattr(s, "text", "")) for s in materialized).strip()
    weighted = 0.0
    wsum = 0.0
    no_speech: list[float] = []
    compression: list[float] = []
    for s in materialized:
        w = max(float(getattr(s, "start", 0.0) or 0.0) - float(getattr(s, "end", 0.0) or 0.0), 0.01)
        w = max(float(getattr(s, "end", 0.0) or 0.0) - float(getattr(s, "start", 0.0) or 0.0), 0.01)
        lp = getattr(s, "avg_logprob", None)
        if lp is not None:
            weighted += float(lp) * w
            wsum += w
        ns = getattr(s, "no_speech_prob", None)
        if ns is not None:
            no_speech.append(float(ns))
        cr = getattr(s, "compression_ratio", None)
        if cr is not None:
            compression.append(float(cr))
    return {
        "path": str(path),
        "lang": lang,
        "text": text,
        "avg_logprob": weighted / wsum if wsum else None,
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "max_no_speech_probability": max(no_speech) if no_speech else None,
        "max_compression_ratio": max(compression) if compression else None,
        "audio_duration": float(getattr(info, "duration", 0.0) or 0.0),
        "segment_count": len(materialized),
    }


def worker(worker_id: int, gpu_id: str, model_size: str, paths: list[str], out_q) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        model = load_model(model_size, "cuda")
        for p in paths:
            out_q.put(("result", worker_id, transcribe_one(model, Path(p))))
    except BaseException:
        out_q.put(("error", worker_id, traceback.format_exc()))
    finally:
        out_q.put(("done", worker_id, None))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--gpus", default="0,1", help="comma separated GPU ids")
    ap.add_argument("--limit", type=int, default=0, help="pilot: transcribe first N zh + N ja")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio-dir", type=Path, default=AUDIO_DIR,
                    help="audio dir to transcribe (default: dataset_clean/audio)")
    ap.add_argument("--out-json", type=Path, default=OUT_JSON,
                    help="output cache json (default: dataset_clean/asr_retranscribed.json)")
    args = ap.parse_args()

    audio_dir = args.audio_dir
    wavs = sorted(audio_dir.glob("*.wav"))
    zh = [p for p in wavs if lang_of(p) == "zh"]
    ja = [p for p in wavs if lang_of(p) == "ja"]
    if args.limit:
        import random

        rng = random.Random(args.seed)
        zh = rng.sample(zh, min(args.limit, len(zh)))
        ja = rng.sample(ja, min(max(args.limit // 2, 1), len(ja)))

    cache: dict[str, dict] = {}
    if args.out_json.is_file():
        cache = {str(x.get("path")): x for x in json.loads(args.out_json.read_text(encoding="utf-8"))}
    # normalize cache keys to basename so dataset_clean results are reused
    cache = {str(Path(k).name): v for k, v in cache.items()}

    jobs = zh + ja
    if cache:
        jobs = [p for p in jobs if p.name not in cache]
        print(f"(reuse {len(cache)} cached records)")
    print(f"jobs: {len(jobs)} (zh={len(zh)} ja={len(ja)}) | gpus={args.gpus}")
    if not jobs:
        print("nothing to transcribe; all cached")
        return 0

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    shards: list[list[str]] = [[] for _ in gpus]
    for i, p in enumerate(jobs):
        shards[i % len(gpus)].append(str(p))

    out_q: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=worker, args=(i, gpu, args.model, shard, out_q), daemon=True)
        for i, (gpu, shard) in enumerate(zip(gpus, shards))
        if shard
    ]
    for p in procs:
        p.start()

    pending = len(jobs)
    done_procs = set()
    try:
        while pending > 0 or len(done_procs) < len(procs):
            kind, wid, payload = out_q.get(timeout=600)
            if kind == "result":
                cache[Path(str(payload["path"])).name] = payload
                pending -= 1
                if pending % 50 == 0:
                    print(f"  progress: {len(jobs) - pending}/{len(jobs)}", flush=True)
            elif kind == "error":
                print(f"worker {wid} error:\n{payload}", file=sys.stderr, flush=True)
            elif kind == "done":
                done_procs.add(wid)
    finally:
        for p in procs:
            p.join(timeout=30)

    args.out_json.write_text(
        json.dumps(list(cache.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"done: {len(cache)} records -> {args.out_json}")

    # comparison report for the just-transcribed subset
    manifest = json.loads((PROJECT_ROOT / "data" / "dataset_clean" / "manifest_clean.json").read_text(encoding="utf-8"))
    old_by_path = {str(x["dataset_path"]): x for x in manifest}
    rows = []
    for p in jobs:
        rec = cache.get(p.name)
        if not rec:
            continue
        old = old_by_path.get(f"audio/{p.name}")
        if old is None:
            continue
        rows.append((p.name, rec, old))
    zh_rows = [r for r in rows if r[0].endswith("_zh.wav")]
    ja_rows = [r for r in rows if r[0].endswith("_jp.wav")]
    for label, rs in (("zh", zh_rows), ("ja", ja_rows)):
        if not rs:
            continue
        old_lp = [r[2]["metrics"]["avg_logprob"] for r in rs if r[2]]
        new_lp = [r[1]["avg_logprob"] for r in rs if r[1]["avg_logprob"] is not None]
        old_med = sorted(old_lp)[len(old_lp) // 2] if old_lp else 0
        new_med = sorted(new_lp)[len(new_lp) // 2] if new_lp else 0
        print(f"\n[{label}] n={len(rs)} avg_logprob 旧中位={old_med:.3f} -> 新中位={new_med:.3f}")
        worse = sum(1 for r in rs if r[1]["avg_logprob"] is not None and r[2] and r[1]["avg_logprob"] > r[2]["metrics"]["avg_logprob"])
        print(f"  变差条数: {worse}/{len(rs)}")
        for name, rec, old in rs[:6]:
            print(f"  {name}: OLD[{old['text'][:40]}] NEW[{rec['text'][:40]}] lp={rec['avg_logprob']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
