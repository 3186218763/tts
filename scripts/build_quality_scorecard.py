"""CLI: build quality_scorecard.json from pipeline stage JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dataset_hq_config import config_to_dict, load_dataset_hq_config  # noqa: E402
from quality_scorecard_lib import build_scorecard_items  # noqa: E402


def _read_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        return list(data["items"])
    if isinstance(data, list):
        return list(data)
    return []


def _load_plan_meta(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        bvid = str(item.get("bvid") or "")
        if bvid:
            out[bvid] = item
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build HQ quality scorecard")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset_hq.yaml")
    p.add_argument("--filter", type=Path, default=PROJECT_ROOT / "data" / "filter_results.json")
    p.add_argument("--speaker", type=Path, default=PROJECT_ROOT / "data" / "speaker_results.json")
    p.add_argument("--asr", type=Path, default=PROJECT_ROOT / "data" / "asr_results.json")
    p.add_argument(
        "--alignment", type=Path, default=PROJECT_ROOT / "data" / "alignment_results.json"
    )
    p.add_argument("--plan", type=Path, default=PROJECT_ROOT / "data" / "collect_plan.json")
    p.add_argument("-o", "--output", type=Path, default=PROJECT_ROOT / "data" / "quality_scorecard.json")
    args = p.parse_args(argv)

    cfg = load_dataset_hq_config(args.config)

    try:
        from build_dataset import canonical_path
    except Exception:  # noqa: BLE001
        canonical_path = None

    items = build_scorecard_items(
        filter_rows=_read_list(args.filter),
        speaker_rows=_read_list(args.speaker),
        asr_rows=_read_list(args.asr),
        align_rows=_read_list(args.alignment),
        bvid_to_meta=_load_plan_meta(args.plan if args.plan.exists() else None),
        cfg=cfg,
        canonical_path=canonical_path,
    )
    doc = {
        "version": 1,
        "scorecard_version": "1",
        "thresholds": config_to_dict(cfg),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for it in items:
        counts[it["decision"]] = counts.get(it["decision"], 0) + 1
    print(f"scorecard items={len(items)} {counts} → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
