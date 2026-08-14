#!/usr/bin/env python3
"""Second-pass independent ASR for zh clips (cross-check vs pass 1).

Uses different decoding params (temperature 0.6, beam 3, no initial prompt)
so agreement with pass 1 is evidence the transcript matches the audio.

Run: /home/mtr/miniconda3/envs/gptsovits/bin/python scripts/retranscribe_pass2.py
Output: data/dataset_hq/asr_pass2_zh.json (list of records, keyed by basename)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "dataset_hq" / "audio"
OUT_JSON = PROJECT_ROOT / "data" / "dataset_hq" / "asr_pass2_zh.json"


def transcribe_one(model, path: Path) -> dict:
    segments, info = model.transcribe(
        str(path),
        language="zh",
        initial_prompt=None,
        beam_size=3,
        best_of=5,
        temperature=0.6,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    materialized = list(segments)
    text = "".join(str(getattr(s, "text", "")) for s in materialized).strip()
    weighted = 0.0
    wsum = 0.0
    for s in materialized:
        w = max(float(getattr(s, "end", 0.0) or 0.0) - float(getattr(s, "start", 0.0) or 0.0), 0.01)
        lp = getattr(s, "avg_logprob", None)
        if lp is not None:
            weighted += float(lp) * w
            wsum += w
    return {
        "path": str(path),
        "lang": "zh",
        "text": text,
        "avg_logprob": weighted / wsum if wsum else None,
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
    }


def worker(worker_id: int, gpu_id: str, model_size: str, paths: list[str], out_q) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        for p in paths:
            out_q.put(("result", worker_id, transcribe_one(model, Path(p))))
    except BaseException:
        out_q.put(("error", worker_id, traceback.format_exc()))
    finally:
        out_q.put(("done", worker_id, None))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    wavs = sorted(p for p in AUDIO_DIR.glob("*_zh.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"pass2 zh jobs: {len(wavs)}")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    shards: list[list[str]] = [[] for _ in gpus]
    for i, p in enumerate(wavs):
        shards[i % len(gpus)].append(str(p))

    out_q: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=worker, args=(i, gpu, args.model, shard, out_q), daemon=True)
        for i, (gpu, shard) in enumerate(zip(gpus, shards))
        if shard
    ]
    for p in procs:
        p.start()

    cache: dict[str, dict] = {}
    if OUT_JSON.is_file():
        cache = {Path(str(x.get("path"))).name: x for x in json.loads(OUT_JSON.read_text(encoding="utf-8"))}
    pending = len(wavs)
    done_procs: set[int] = set()
    try:
        while pending > 0 or len(done_procs) < len(procs):
            kind, wid, payload = out_q.get(timeout=600)
            if kind == "result":
                cache[Path(str(payload["path"])).name] = payload
                pending -= 1
                if pending % 100 == 0:
                    print(f"  progress: {len(wavs) - pending}/{len(wavs)}", flush=True)
            elif kind == "error":
                print(f"worker {wid} error:\n{payload}", file=sys.stderr, flush=True)
            elif kind == "done":
                done_procs.add(wid)
    finally:
        for p in procs:
            p.join(timeout=30)

    OUT_JSON.write_text(
        json.dumps(list(cache.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"done: {len(cache)} records -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
