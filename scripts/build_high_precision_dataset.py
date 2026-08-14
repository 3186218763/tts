#!/usr/bin/env python3
"""Build a no-human high-precision subset from dual-ASR consensus.

Reads:
  data/dataset_hq/annotation.list
  data/dataset_hq/asr_consensus_whisper.json
  data/dataset_hq/asr_consensus_funasr.json
  optional: data/dataset_hq/asr_consensus_third.json  (soft-band third vote)

Writes:
  data/dataset_precision/annotation.list
  data/dataset_precision/manifest.json
  data/dataset_precision/report.json
  data/dataset_precision/dropped.jsonl
  data/dataset_precision/audio/  (hardlinks)

Policy (default thresholds in consensus_transcript_lib):
  keep if Whisper↔FunASR sim >= 0.92 (strict)
  soft band [0.85, 0.92) only if third ASR agrees
  else drop

This optimizes for **label precision**, not coverage. Expected keep rate on
noisy live speech is often 40–70% of zh clips.

Run:
  python scripts/build_high_precision_dataset.py
  python scripts/build_high_precision_dataset.py --apply-links
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import consensus_transcript_lib as ctl  # noqa: E402
import dataset_text_quality as tq  # noqa: E402

HQ = PROJECT_ROOT / "data" / "dataset_hq"
OUT = PROJECT_ROOT / "data" / "dataset_precision"
ANNOTATION = HQ / "annotation.list"
WHISPER = HQ / "asr_consensus_whisper.json"
FUNASR = HQ / "asr_consensus_funasr.json"
THIRD = HQ / "asr_consensus_third.json"
AUDIO_SRC = HQ / "audio"


def load_engine_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        out = {}
        for row in data:
            name = row.get("name") or Path(str(row.get("path", ""))).name
            out[name] = row
        return out
    return data


def load_annotation() -> list[dict]:
    rows = []
    for line in ANNOTATION.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path, spk, lang, text = line.split("|", 3)
        name = path.split("/")[-1]
        lang_n = "ja" if lang.upper() in {"JP", "JA"} else "zh"
        rows.append(
            {
                "name": name,
                "path": path,
                "spk": spk,
                "lang_code": lang.upper() if lang.upper() in {"ZH", "JP"} else ("JP" if lang_n == "ja" else "ZH"),
                "lang": lang_n,
                "old_text": text,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["zh", "ja", "all"], default="all")
    ap.add_argument("--keep-strict", type=float, default=0.92)
    ap.add_argument("--keep-soft", type=float, default=0.85)
    ap.add_argument("--third-agree", type=float, default=0.92)
    ap.add_argument(
        "--fallback-same-family",
        action="store_true",
        help="if FunASR missing for a clip, fall back to old annotation text "
        "as secondary (weaker; not recommended for 98%% proxy)",
    )
    ap.add_argument(
        "--require-funasr",
        action="store_true",
        default=True,
        help="drop clips without FunASR (default on)",
    )
    ap.add_argument("--no-require-funasr", action="store_false", dest="require_funasr")
    ap.add_argument("--apply-links", action="store_true", help="hardlink audio into out dir")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    thr = ctl.ConsensusThresholds(
        keep_strict=args.keep_strict,
        keep_soft=args.keep_soft,
        third_agree=args.third_agree,
    )

    ann = load_annotation()
    if args.lang != "all":
        ann = [r for r in ann if r["lang"] == args.lang]

    whisper = load_engine_map(WHISPER)
    funasr = load_engine_map(FUNASR)
    third_map = load_engine_map(THIRD)

    decisions: list[dict] = []
    kept_lines: list[str] = []
    kept_manifest: list[dict] = []
    dropped_lines: list[str] = []

    for row in ann:
        name = row["name"]
        lang = row["lang"]
        w = whisper.get(name)
        f = funasr.get(name)
        t = third_map.get(name)

        primary = (w or {}).get("text") if w else None
        # Prefer heterogeneous secondary; optional fallback to old label
        if f and (f.get("text") or "").strip():
            secondary = f.get("text")
            secondary_src = "funasr"
        elif args.fallback_same_family:
            secondary = row["old_text"]
            secondary_src = "old_annotation"
        else:
            secondary = None
            secondary_src = None

        if args.require_funasr and secondary_src != "funasr":
            dec = ctl.ConsensusDecision(
                decision="drop",
                tier="drop_no_funasr",
                text=None,
                sim_ab=None,
                sim_ac=None,
                sim_bc=None,
                cer_ab=None,
                source=None,
            )
        else:
            # If no whisper, try funasr vs old annotation only when fallback on
            if primary is None:
                if secondary is not None and args.fallback_same_family:
                    primary = row["old_text"]
                else:
                    dec = ctl.ConsensusDecision(
                        decision="drop",
                        tier="drop_no_whisper",
                        text=None,
                        sim_ab=None,
                        sim_ac=None,
                        sim_bc=None,
                        cer_ab=None,
                        source=None,
                    )
                    primary = None

            if primary is not None and secondary is not None:
                third_text = (t or {}).get("text") if t else None
                dec = ctl.adjudicate(
                    primary,
                    secondary,
                    third=third_text,
                    lang=lang,
                    thresholds=thr,
                    primary_logprob=(w or {}).get("avg_logprob"),
                )
            elif primary is None:
                pass  # dec already set
            else:
                dec = ctl.ConsensusDecision(
                    decision="drop",
                    tier="drop_no_pair",
                    text=None,
                    sim_ab=None,
                    sim_ac=None,
                    sim_bc=None,
                    cer_ab=None,
                    source=None,
                )

        # text quality gate on kept text
        if dec.decision == "keep" and dec.text:
            tqr = tq.evaluate_text_quality(dec.text, lang)
            if not tqr.keep:
                dec = ctl.ConsensusDecision(
                    decision="drop",
                    tier="drop_text_quality",
                    text=None,
                    sim_ab=dec.sim_ab,
                    sim_ac=dec.sim_ac,
                    sim_bc=dec.sim_bc,
                    cer_ab=dec.cer_ab,
                    source=None,
                )

        item = {
            "name": name,
            "lang": lang,
            "old_text": row["old_text"],
            "whisper_text": (w or {}).get("text"),
            "funasr_text": (f or {}).get("text"),
            "secondary_src": secondary_src,
            **dec.to_dict(),
        }
        decisions.append(item)

        if dec.decision == "keep" and dec.text:
            line = f"audio/{name}|{row['spk']}|{row['lang_code']}|{dec.text}"
            kept_lines.append(line)
            kept_manifest.append(
                {
                    "dataset_path": f"audio/{name}",
                    "lang": lang,
                    "text": dec.text,
                    "tier": dec.tier,
                    "sim_ab": dec.sim_ab,
                    "source": dec.source,
                    "text_source": "consensus",
                }
            )
        else:
            dropped_lines.append(json.dumps(item, ensure_ascii=False))

    summary = ctl.summarize_decisions(decisions)
    by_lang: dict[str, dict] = {}
    for lang in sorted({r["lang"] for r in decisions}):
        subset = [r for r in decisions if r["lang"] == lang]
        by_lang[lang] = ctl.summarize_decisions(subset)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotation.list").write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(kept_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "dropped.jsonl").write_text(
        "\n".join(dropped_lines) + ("\n" if dropped_lines else ""), encoding="utf-8"
    )
    report = {
        "thresholds": {
            "keep_strict": thr.keep_strict,
            "keep_soft": thr.keep_soft,
            "third_agree": thr.third_agree,
        },
        "summary": summary,
        "by_lang": by_lang,
        "whisper_cache": str(WHISPER),
        "funasr_cache": str(FUNASR),
        "n_whisper": len(whisper),
        "n_funasr": len(funasr),
        "policy": (
            "No-human high-precision: keep only multi-ASR consensus. "
            "Inter-ASR agreement is a precision proxy, not human CER. "
            "For TTS fine-tune, fewer clean labels beat more noisy ones."
        ),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.apply_links:
        audio_out = out_dir / "audio"
        audio_out.mkdir(parents=True, exist_ok=True)
        linked = 0
        for m in kept_manifest:
            name = Path(m["dataset_path"]).name
            src = AUDIO_SRC / name
            dst = audio_out / name
            if not src.is_file():
                print(f"WARN missing audio: {src}", flush=True)
                continue
            if dst.exists() or dst.is_symlink():
                linked += 1
                continue
            try:
                os.link(src, dst)
            except OSError:
                dst.symlink_to(src)
            linked += 1
        print(f"linked audio: {linked}", flush=True)

    print(
        f"keep={summary['n_keep']}/{summary['n_total']} "
        f"({summary['keep_rate']*100:.1f}%) "
        f"identical_proxy={summary.get('keep_identical_rate')} "
        f"-> {out_dir}",
        flush=True,
    )
    for lang, s in by_lang.items():
        print(
            f"  {lang}: keep={s['n_keep']}/{s['n_total']} "
            f"({s['keep_rate']*100:.1f}%) tiers={s['tiers']}",
            flush=True,
        )
    if summary["n_keep"] == 0:
        print(
            "NOTE: no keeps yet — run scripts/run_consensus_asr.py first "
            "to fill dual-ASR caches.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
