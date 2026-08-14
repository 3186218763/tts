#!/usr/bin/env python3
"""LLM-correct ASR transcripts for the curated dataset (minimal edits only).

Fixes fillers, stutters, obvious ASR slips and punctuation, while a
similarity guard (SequenceMatcher >= min_sim) rejects any rewrite that
drifts too far from the original — those keep the raw ASR text.

Input : data/dataset_clean/asr_retranscribed.json
Output: data/dataset_clean/text_corrected.json   {path: corrected_text}
        data/dataset_clean/correct_rejected.json {path: original_text}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASR_JSON = PROJECT_ROOT / "data" / "dataset_clean" / "asr_retranscribed.json"
OUT_JSON = PROJECT_ROOT / "data" / "dataset_clean" / "text_corrected.json"
REJECT_JSON = PROJECT_ROOT / "data" / "dataset_clean" / "correct_rejected.json"

BATCH = 25
MIN_SIM = 0.72

SYSTEM_PROMPT = (
    "你是音频转录校对员。下面是从中文直播语音转写的句子列表，可能包含："
    "语气词残留（嗯/啊/诶/哦/呃）、重复（谢谢谢谢）、口胡错字、标点缺失、句首尾截断。\n"
    "要求：1) 只做最小修正：删明显语气词和重复、规范标点、修正明显同音错字；"
    "2) 禁止改写原意、禁止编造或补全原文没有的内容；3) 不确定的句子原样保留；"
    "4) 输出必须是 JSON，格式 {\"results\":[{\"index\":0,\"text\":\"...\"}]}，"
    "与输入顺序一一对应，index 必须完整。"
)


def load_config() -> dict:
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    return raw["llm"]


def call_batch(api_key: str, base_url: str, model: str, items: list[str]) -> list[dict]:
    lines = "\n".join(f"{i}. {text}" for i, text in enumerate(items))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请校对以下 {len(items)} 条转录：\n{lines}"},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # strip code fences if the model wraps JSON
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    results = json.loads(content)["results"]
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only correct first N zh clips")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip paths already present in text_corrected.json / correct_rejected.json")
    args = ap.parse_args()

    records = json.loads(ASR_JSON.read_text(encoding="utf-8"))
    zh = [r for r in records if r["lang"] == "zh"]
    zh.sort(key=lambda r: r["path"])
    if args.limit:
        zh = zh[: args.limit]
    print(f"zh transcripts to correct: {len(zh)}")

    cfg = load_config()
    api_key, base_url, model = cfg["api_key"], cfg["base_url"], cfg["model"]

    corrected: dict[str, str] = {}
    rejected: dict[str, str] = {}
    if OUT_JSON.is_file():
        corrected = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if REJECT_JSON.is_file():
        rejected = json.loads(REJECT_JSON.read_text(encoding="utf-8"))

    if args.only_missing:
        done_paths = set(corrected) | set(rejected)
        zh = [r for r in zh if r["path"] not in done_paths]
        print(f"only-missing: {len(zh)} clips remaining")
        if not zh:
            return 0

    batches = [zh[i : i + args.batch] for i in range(0, len(zh), args.batch)]
    done = 0
    fails = 0

    def process(batch: list[dict]) -> tuple[list[tuple[str, str, bool]], int]:
        items = [{"index": i, "text": r["text"]} for i, r in enumerate(batch)]
        for attempt in range(6):
            try:
                results = call_batch(api_key, base_url, model, [x["text"] for x in items])
                idx_map = {int(r["index"]): r["text"] for r in results if "index" in r}
                out = []
                for pos, rec in enumerate(batch):
                    new = idx_map.get(pos, rec["text"])
                    sim = SequenceMatcher(None, rec["text"], new).ratio()
                    keep_new = new.strip() and sim >= MIN_SIM
                    out.append((rec["path"], new if keep_new else rec["text"], keep_new))
                return out, 0
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    print(f"  batch fail: {type(exc).__name__}: {exc}", flush=True)
                if attempt == 5:
                    return [], 1
                time.sleep(3 * (attempt + 1))
        return [], 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, b) for b in batches]
        for fut in as_completed(futures):
            out, fail = fut.result()
            fails += fail
            for path, text, ok in out:
                (corrected if ok else rejected)[path] = text
            done += 1
            if done % 10 == 0:
                print(f"  batches done: {done}/{len(batches)} (fails={fails})", flush=True)
            if done % 25 == 0:
                OUT_JSON.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                REJECT_JSON.write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    OUT_JSON.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REJECT_JSON.write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done: corrected={len(corrected)} rejected_kept_orig={len(rejected)} fails={fails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
