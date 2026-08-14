#!/usr/bin/env python3
"""Export curated clips from data/dataset_hq into data/dataset_clean/.

- hard-links selected wav files (no extra disk usage)
- writes GPT-SoVITS style annotation.list (ZH/JP)
- writes manifest_clean.json + sample audit list (50 zh + 20 ja)
- original data/dataset_hq is never modified
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "data" / "dataset_hq"
CURATED = SRC / "curated_manifest.json"
OUT = PROJECT_ROOT / "data" / "dataset_clean"
ANNOTATION = OUT / "annotation.list"
SAMPLE_ZHS = 50
SAMPLE_JAS = 20

LANG_CODE = {"zh": "ZH", "ja": "JP"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zh-sample", type=int, default=SAMPLE_ZHS)
    ap.add_argument("--ja-sample", type=int, default=SAMPLE_JAS)
    args = ap.parse_args()

    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    print(f"curated: {len(curated)} clips")

    audio_out = OUT / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    ann_lines: list[str] = []
    manifest_out: list[dict] = []
    for item in curated:
        lang = item["lang"]
        src_wav = SRC / str(item["dataset_path"])
        if not src_wav.is_file():
            print(f"WARN missing: {src_wav}")
            continue
        name = Path(str(item["dataset_path"])).name
        dst = audio_out / name
        if not dst.exists():
            try:
                os.link(src_wav, dst)
            except OSError:
                dst.symlink_to(src_wav)
        ann_lines.append(f"audio/{name}|花音|{LANG_CODE[lang]}|{item['text']}")
        manifest_out.append(item)

    ANNOTATION.write_text("\n".join(ann_lines) + "\n", encoding="utf-8")
    (OUT / "manifest_clean.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    zh = [x for x in manifest_out if x["lang"] == "zh"]
    ja = [x for x in manifest_out if x["lang"] == "ja"]
    print(f"exported: zh={len(zh)} ja={len(ja)} total={len(manifest_out)} -> {OUT}")
    print("tier zh:", dict(Counter(x["tier"] for x in zh)))
    print("tier ja:", dict(Counter(x["tier"] for x in ja)))

    rng = random.Random(args.seed)
    sample = rng.sample(zh, min(args.zh_sample, len(zh))) + rng.sample(ja, min(args.ja_sample, len(ja)))
    lines = [
        "# 精选数据集人工抽查清单（文本来自 ASR，请听音频核对）",
        f"# zh 抽样 {len([s for s in sample if s['lang']=='zh'])} 条 / ja 抽样 {len([s for s in sample if s['lang']=='ja'])} 条",
        "# 格式: dataset_path | tier | text | duration | logprob | unk/flags",
    ]
    for s in sample:
        flags = "/".join(s.get("flags") or []) or "-"
        dur = s.get("metrics", {}).get("duration", 0.0)
        lp = s.get("metrics", {}).get("avg_logprob", 0.0)
        lines.append(
            f"{s['dataset_path']} | {s['tier']} | {s['text']} | {dur:.1f}s | "
            f"{lp:.2f} | {flags}"
        )
    sample_path = OUT / "audit_sample.txt"
    sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audit sample: {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
