#!/usr/bin/env python3
"""Analyze HQ dataset quality (zh + ja) before curated cleanup.

Reads data/dataset_hq/manifest.json plus GPT-SoVITS 2-name2text.txt
(phonemes / UNK / BERT) and reports, per language, how many clips each
quality rule would drop:

  - text quality gates (scripts/dataset_text_quality.py)
  - phoneme length too short (<12) / too long (>60)
  - audio duration outside 2.5-10 s
  - ASR confidence: avg_logprob / language_probability / comp.asr
  - truncated sentence heuristic (short + no end punctuation)
  - ja: UNK phoneme tokens, missing BERT features

Output: data/dataset_hq/quality_analysis.json (per-clip + summary).
Run with --apply to also write data/dataset_hq/curated_manifest.json
(clips passing every gate; original files are never modified).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dataset_text_quality as tq

MANIFEST = PROJECT_ROOT / "data" / "dataset_hq" / "manifest.json"
PHONEME_TXT = Path("/home/mtr/tt/GPT-SoVITS/logs/huayin-hq/2-name2text.txt")
OUT_JSON = PROJECT_ROOT / "data" / "dataset_hq" / "quality_analysis.json"
OUT_CURATED = PROJECT_ROOT / "data" / "dataset_hq" / "curated_manifest.json"

PHONEME_MIN = 12
PHONEME_MAX = 60
DUR_MIN = 2.5
DUR_MAX = 10.0
LOGPROB_MAX = {"zh": -0.45, "ja": -0.30}  # whisper avg_logprob (<= is better)
TRUNC_MIN_CHARS = {"zh": 8, "ja": 12}
LANG_PROB_MIN = 0.70
ASR_COMP_MIN = 0.50

_END_PUNCT_RE = re.compile(r"[。！？…!?.;；]$")


def has_end_punct(text: str) -> bool:
    return bool(_END_PUNCT_RE.search((text or "").strip()))


def is_truncated(text: str, *, min_chars: int) -> tuple[bool, str]:
    """Short sentence with no end punctuation -> likely ASR cut-off."""
    t = (text or "").strip()
    if len(t) < min_chars and not has_end_punct(t):
        return True, "truncated_short"
    return False, ""


def load_phonemes(path: Path) -> dict[str, str]:
    phonemes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            phonemes[parts[0]] = parts[1]
    return phonemes


def evaluate_clip(
    item: dict,
    phonemes: dict[str, str],
    lang: str,
    *,
    logprob_max: float,
    trunc_min_chars: int,
) -> dict:
    name = str(item.get("dataset_path") or "").split("/")[-1]
    phone = phonemes.get(name, "")
    n_ph = len(phone.split()) if phone else 0
    n_unk = phone.count("UNK")
    text = str(item.get("text") or "").strip()
    metrics = item.get("metrics") or {}
    comp = item.get("components") or {}
    duration = float(metrics.get("duration") or 0.0)
    avg_logprob = float(metrics.get("avg_logprob") or 0.0)
    lang_prob = float(metrics.get("language_probability") or 0.0)
    asr_comp = float(comp.get("asr") or 0.0)
    align_comp = float(comp.get("align") or 0.0)

    tqr = tq.evaluate_text_quality(text, lang)
    trunc, trunc_reason = is_truncated(text, min_chars=trunc_min_chars)

    reasons: list[str] = []
    if not tqr.keep:
        reasons.extend(f"text:{r}" for r in tqr.reasons)
    if n_ph and n_ph < PHONEME_MIN:
        reasons.append(f"phoneme_lt{PHONEME_MIN}")
    if n_ph and n_ph > PHONEME_MAX:
        reasons.append(f"phoneme_gt{PHONEME_MAX}")
    if not (DUR_MIN <= duration <= DUR_MAX):
        reasons.append("duration_out_of_range")
    if avg_logprob > logprob_max:
        reasons.append(f"logprob_gt{-logprob_max}")
    if lang_prob < LANG_PROB_MIN:
        reasons.append("lang_prob_low")
    if asr_comp < ASR_COMP_MIN:
        reasons.append("asr_comp_low")
    if trunc:
        reasons.append(trunc_reason)
    flags: list[str] = []
    if lang == "ja":
        if n_unk > 0:
            flags.append("ja_unk_phoneme")
        if phone and not Path(
            f"/home/mtr/tt/GPT-SoVITS/logs/huayin-hq/3-bert/{name}.pt"
        ).is_file():
            flags.append("ja_bert_missing")

    return {
        "dataset_path": item.get("dataset_path"),
        "lang": lang,
        "tier": item.get("tier"),
        "source_bvid": item.get("source_bvid"),
        "text": text,
        "duration": duration,
        "phoneme_len": n_ph,
        "unk_count": n_unk,
        "avg_logprob": avg_logprob,
        "language_probability": lang_prob,
        "comp_asr": asr_comp,
        "comp_align": align_comp,
        "score": item.get("score"),
        "truncated": trunc,
        "reasons": reasons,
        "keep": not reasons,
        "flags": flags,
    }


def summarize(records: list[dict], lang: str) -> dict:
    total = len(records)
    keep = [r for r in records if r["keep"]]
    dropped = [r for r in records if not r["keep"]]
    reasons: Counter[str] = Counter()
    for r in dropped:
        reasons.update(r["reasons"])
    return {
        "lang": lang,
        "total": total,
        "keep": len(keep),
        "drop": len(dropped),
        "drop_pct": round(len(dropped) / total * 100, 1) if total else 0.0,
        "keep_tier": dict(Counter(r["tier"] for r in keep)),
        "drop_reasons": dict(reasons.most_common()),
        "keep_by_reason": {k: v for k, v in reasons.most_common()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="also write curated_manifest.json")
    ap.add_argument("--logprob-zh", type=float, default=LOGPROB_MAX["zh"],
                    help=f"zh avg_logprob cutoff (<= is good; default {LOGPROB_MAX['zh']})")
    ap.add_argument("--logprob-ja", type=float, default=LOGPROB_MAX["ja"],
                    help=f"ja avg_logprob cutoff (<= is good; default {LOGPROB_MAX['ja']})")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    phonemes = load_phonemes(PHONEME_TXT)
    print(f"manifest: {len(manifest)} clips | phoneme table: {len(phonemes)}")

    records: list[dict] = []
    for item in manifest:
        lang = tq.normalize_lang(item.get("lang"))
        if lang not in {"zh", "ja"}:
            continue
        lp_max = args.logprob_zh if lang == "zh" else args.logprob_ja
        rec = evaluate_clip(
            item, phonemes, lang,
            logprob_max=lp_max,
            trunc_min_chars=TRUNC_MIN_CHARS[lang],
        )
        records.append(rec)

    summaries = [summarize([r for r in records if r["lang"] == lang], lang)
                 for lang in ("zh", "ja")]

    print("\n=== 质量分析摘要 ===")
    for s in summaries:
        print(f"\n[{s['lang']}] total={s['total']} keep={s['keep']} drop={s['drop']} ({s['drop_pct']}%)")
        print(f"  keep tier: {s['keep_tier']}")
        print("  drop reasons:")
        for reason, n in s["drop_reasons"].items():
            print(f"    {reason}: {n}")

    payload = {
        "summaries": summaries,
        "flags": {
            "zh": {"ja_unk_phoneme": 0, "ja_bert_missing": 0},
            "ja": {
                "ja_unk_phoneme": sum(1 for r in records if r["lang"] == "ja" and "ja_unk_phoneme" in r["flags"]),
                "ja_bert_missing": sum(1 for r in records if r["lang"] == "ja" and "ja_bert_missing" in r["flags"]),
            },
        },
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n分析结果已写入: {args.out}")

    if args.apply:
        curated = [r for r in records if r["keep"]]
        manifest_by_path = {str(x.get("dataset_path")): x for x in manifest}
        curated_full = [manifest_by_path[str(r["dataset_path"])] | {"reasons": r["reasons"]}
                        for r in curated]
        OUT_CURATED.write_text(
            json.dumps(curated_full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"精选清单已写入: {OUT_CURATED} ({len(curated_full)} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
