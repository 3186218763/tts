#!/usr/bin/env python3
"""Cross-check pass-1 vs pass-2 ASR for zh clips; flag low-agreement ones.

Agreement (SequenceMatcher ratio on compact text) >= threshold means the
transcript is trustworthy (audio really says it, incl. genuine stutters).
Low agreement -> likely ASR mishearing; those are dropped from training.

Output:
  data/dataset_hq/crosscheck_report.json (per-clip agreement + decision)
  data/dataset_hq/suspicious_zh.txt   (low-agreement list for human review)
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HQ = PROJECT_ROOT / "data" / "dataset_hq"
PASS1 = PROJECT_ROOT / "data" / "dataset_clean" / "asr_retranscribed.json"
PASS2 = HQ / "asr_pass2_zh.json"
REPORT = HQ / "crosscheck_report.json"
SUSPICIOUS = HQ / "suspicious_zh.txt"

AGREE_MIN = 0.85

_PUNCT_RE = re.compile(r"[\s，。！？、；：…,.!?;:·~～\-_/\\|()（）「」『』【】\"'“”]+")


def compact(text: str) -> str:
    try:
        from zhconv import convert

        text = convert(text or "", "zh-cn")
    except Exception:
        pass
    return _PUNCT_RE.sub("", text or "")


def main() -> int:
    pass1 = {Path(r["path"]).name: r for r in json.loads(PASS1.read_text(encoding="utf-8"))}
    pass2 = {Path(r["path"]).name: r for r in json.loads(PASS2.read_text(encoding="utf-8"))}
    print(f"pass1 zh: {sum(1 for k in pass1 if k.endswith('_zh.wav'))} | pass2: {len(pass2)}")

    rows = []
    low = []
    for name, r2 in pass2.items():
        r1 = pass1.get(name)
        if r1 is None:
            rows.append({"name": name, "text1": "", "text2": r2["text"],
                         "sim": 0.0, "agree": False, "note": "no_pass1"})
            low.append((name, "", r2["text"], 0.0))
            continue
        a, b = compact(r1["text"]), compact(r2["text"])
        if not a and not b:
            sim = 1.0
        elif not a or not b:
            sim = 0.0
        else:
            sim = SequenceMatcher(None, a, b).ratio()
        agree = sim >= AGREE_MIN
        rows.append({"name": name, "text1": r1["text"], "text2": r2["text"],
                     "sim": round(sim, 4), "agree": agree})
        if not agree:
            low.append((name, r1["text"], r2["text"], sim))

    REPORT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# zh 双ASR不一致清单（建议剔除/复核）", f"# 阈值 sim>={AGREE_MIN} 为一致 | 共 {len(low)} 条\n"]
    for name, t1, t2, sim in sorted(low, key=lambda x: x[3]):
        lines.append(f"{name} | sim={sim:.2f}\n  P1: {t1}\n  P2: {t2}")
    SUSPICIOUS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_agree = sum(1 for r in rows if r["agree"])
    print(f"agree: {n_agree}/{len(rows)} ({n_agree/len(rows)*100:.1f}%)")
    print(f"low-agreement (drop candidates): {len(low)} -> {SUSPICIOUS}")
    import statistics as st
    sims = [r["sim"] for r in rows]
    print(f"sim dist: p10={sorted(sims)[len(sims)//10]:.3f} med={st.median(sims):.3f} p90={sorted(sims)[len(sims)*9//10]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
