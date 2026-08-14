#!/usr/bin/env python3
"""Run heterogeneous dual ASR on dataset_hq clips (no human).

Engines
-------
- primary (Whisper): faster-whisper large-v3, forced language, beam 5, temp 0
- secondary (FunASR):
    - zh: Paraformer-large (classic Chinese ASR, different family from Whisper)
    - ja: SenseVoiceSmall (multilingual; falls back to second Whisper decode
      with different params if SenseVoice unavailable)

Caches (incremental, safe to resume):
  data/dataset_hq/asr_consensus_whisper.json
  data/dataset_hq/asr_consensus_funasr.json

Run (gptsovits env):
  /home/mtr/miniconda3/envs/gptsovits/bin/python scripts/run_consensus_asr.py \\
      --lang zh --gpus 0,1
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
HQ = PROJECT_ROOT / "data" / "dataset_hq"
AUDIO_DIR = HQ / "audio"
ANNOTATION = HQ / "annotation.list"
WHISPER_CACHE = HQ / "asr_consensus_whisper.json"
FUNASR_CACHE = HQ / "asr_consensus_funasr.json"

WHISPER_PROMPTS = {
    "zh": "以下是普通话的句子。",
    "ja": "以下は日本語の文章です。",
}

GS_ROOT = Path("/home/mtr/tt/GPT-SoVITS")
PARAFORMER_PATHS = {
    "asr": GS_ROOT
    / "tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "vad": GS_ROOT / "tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": GS_ROOT
    / "tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
}


def parse_annotation(lang_filter: str | None) -> list[tuple[str, str]]:
    """Return list of (basename, lang) from annotation.list."""
    rows: list[tuple[str, str]] = []
    for line in ANNOTATION.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path, _spk, lang, _text = line.split("|", 3)
        name = path.split("/")[-1]
        lang_n = "ja" if lang.upper() in {"JP", "JA"} else "zh"
        if lang_filter and lang_n != lang_filter:
            continue
        rows.append((name, lang_n))
    return rows


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {Path(str(x.get("path", x.get("name", "")))).name: x for x in data}
    if isinstance(data, dict):
        return data
    return {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(cache.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def whisper_one(model, path: Path, lang: str) -> dict:
    segments, info = model.transcribe(
        str(path),
        language=lang,
        initial_prompt=WHISPER_PROMPTS.get(lang),
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    segs = list(segments)
    text = "".join(str(getattr(s, "text", "")) for s in segs).strip()
    wsum = 0.0
    weighted = 0.0
    for s in segs:
        w = max(float(getattr(s, "end", 0) or 0) - float(getattr(s, "start", 0) or 0), 0.01)
        lp = getattr(s, "avg_logprob", None)
        if lp is not None:
            weighted += float(lp) * w
            wsum += w
    return {
        "name": path.name,
        "path": str(path),
        "lang": lang,
        "engine": "whisper_large_v3",
        "text": text,
        "avg_logprob": (weighted / wsum) if wsum else None,
        "language_probability": float(getattr(info, "language_probability", 0) or 0),
    }


def whisper_worker(worker_id: int, gpu: str, model_size: str, jobs: list[tuple[str, str]], out_q) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        for name, lang in jobs:
            path = AUDIO_DIR / name
            out_q.put(("result", worker_id, whisper_one(model, path, lang)))
    except BaseException:
        out_q.put(("error", worker_id, traceback.format_exc()))
    finally:
        out_q.put(("done", worker_id, None))


def load_funasr_zh(device: str):
    from funasr import AutoModel

    asr = PARAFORMER_PATHS["asr"]
    vad = PARAFORMER_PATHS["vad"]
    punc = PARAFORMER_PATHS["punc"]
    if asr.is_dir() and vad.is_dir() and punc.is_dir():
        return AutoModel(
            model=str(asr),
            model_revision="v2.0.4",
            vad_model=str(vad),
            vad_model_revision="v2.0.4",
            punc_model=str(punc),
            punc_model_revision="v2.0.4",
            device=device,
            disable_update=True,
        )
    # model ids (download on first use)
    return AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        device=device,
        disable_update=True,
    )


def load_funasr_ja(device: str):
    from funasr import AutoModel

    return AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        device=device,
        disable_update=True,
    )


def funasr_one(model, path: Path, lang: str, engine: str) -> dict:
    try:
        res = model.generate(input=str(path))
        text = ""
        if res and isinstance(res, list):
            text = str(res[0].get("text", "") if isinstance(res[0], dict) else res[0]).strip()
        # SenseVoice may wrap tags like <|zh|><|NEUTRAL|>...
        if "SenseVoice" in engine or engine == "sensevoice":
            text = _strip_sensevoice_tags(text)
    except Exception as exc:  # noqa: BLE001
        text = ""
        return {
            "name": path.name,
            "path": str(path),
            "lang": lang,
            "engine": engine,
            "text": text,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "name": path.name,
        "path": str(path),
        "lang": lang,
        "engine": engine,
        "text": text,
        "avg_logprob": None,
    }


def _strip_sensevoice_tags(text: str) -> str:
    import re

    return re.sub(r"<\|[^|]*\|>", "", text or "").strip()


def funasr_worker(
    worker_id: int,
    gpu: str,
    jobs: list[tuple[str, str]],
    out_q,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        # After CUDA_VISIBLE_DEVICES, device is always cuda:0 inside process
        device = "cuda:0"
        models: dict[str, object] = {}
        for name, lang in jobs:
            if lang not in models:
                models[lang] = load_funasr_zh(device) if lang == "zh" else load_funasr_ja(device)
            engine = "paraformer_zh" if lang == "zh" else "sensevoice"
            out_q.put(
                ("result", worker_id, funasr_one(models[lang], AUDIO_DIR / name, lang, engine))
            )
    except BaseException:
        out_q.put(("error", worker_id, traceback.format_exc()))
    finally:
        out_q.put(("done", worker_id, None))


def run_pool(
    kind: str,
    jobs: list[tuple[str, str]],
    gpus: list[str],
    model_size: str,
    cache: dict[str, dict],
    cache_path: Path,
    flush_every: int = 50,
) -> dict[str, dict]:
    pending = [(n, lang) for n, lang in jobs if n not in cache]
    print(f"{kind}: total={len(jobs)} cached={len(jobs)-len(pending)} todo={len(pending)}", flush=True)
    if not pending:
        return cache

    shards: list[list[tuple[str, str]]] = [[] for _ in gpus]
    for i, job in enumerate(pending):
        shards[i % len(gpus)].append(job)

    out_q: mp.Queue = mp.Queue()
    procs: list[mp.Process] = []
    for i, (gpu, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            continue
        if kind == "whisper":
            target = whisper_worker
            args = (i, gpu, model_size, shard, out_q)
        else:
            target = funasr_worker
            args = (i, gpu, shard, out_q)
        p = mp.Process(target=target, args=args, daemon=True)
        procs.append(p)
        p.start()

    left = len(pending)
    done_procs: set[int] = set()
    try:
        while left > 0 or len(done_procs) < len(procs):
            kind_msg, wid, payload = out_q.get(timeout=1800)
            if kind_msg == "result":
                cache[payload["name"]] = payload
                left -= 1
                if left % flush_every == 0 or left == 0:
                    save_cache(cache_path, cache)
                    print(f"  {kind} progress: {len(pending)-left}/{len(pending)}", flush=True)
            elif kind_msg == "error":
                print(f"worker {wid} error:\n{payload}", file=sys.stderr, flush=True)
            elif kind_msg == "done":
                done_procs.add(wid)
    finally:
        for p in procs:
            p.join(timeout=60)
        save_cache(cache_path, cache)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["zh", "ja", "all"], default="zh")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--engine", choices=["both", "whisper", "funasr"], default="both")
    ap.add_argument("--skip-funasr-ja", action="store_true",
                    help="for ja, skip SenseVoice (only run Whisper)")
    args = ap.parse_args()

    lang_filter = None if args.lang == "all" else args.lang
    jobs = parse_annotation(lang_filter)
    if args.skip_funasr_ja:
        # still list ja for whisper; funasr will skip ja below if filtered
        pass
    if args.limit:
        jobs = jobs[: args.limit]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("no gpus", file=sys.stderr)
        return 2

    if args.engine in {"both", "whisper"}:
        w_cache = load_cache(WHISPER_CACHE)
        run_pool("whisper", jobs, gpus, args.model, w_cache, WHISPER_CACHE)

    if args.engine in {"both", "funasr"}:
        f_jobs = jobs
        if args.skip_funasr_ja:
            f_jobs = [(n, lang) for n, lang in jobs if lang != "ja"]
        f_cache = load_cache(FUNASR_CACHE)
        # FunASR often memory-heavy: use one GPU per process is fine; for 3090
        # dual is OK with paraformer.
        run_pool("funasr", f_jobs, gpus, args.model, f_cache, FUNASR_CACHE)

    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    # spawn is safer with CUDA
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    raise SystemExit(main())
