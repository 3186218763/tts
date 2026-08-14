"""CLI: select HQ subset and write data/dataset_hq/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dataset_hq_config import config_to_dict, load_dataset_hq_config  # noqa: E402
from hq_dataset_lib import select_hq_items, write_hq_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build HQ training dataset with ZH quota")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset_hq.yaml")
    p.add_argument(
        "--scorecard",
        type=Path,
        default=PROJECT_ROOT / "data" / "quality_scorecard.json",
    )
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "dataset_hq")
    p.add_argument("--target-n", type=int, default=None)
    p.add_argument("--zh-ratio", type=float, default=None)
    p.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    args = p.parse_args(argv)

    cfg = load_dataset_hq_config(args.config)
    n_target = args.target_n if args.target_n is not None else cfg.n_target
    zh_ratio = args.zh_ratio if args.zh_ratio is not None else cfg.zh_ratio

    doc = json.loads(args.scorecard.read_text(encoding="utf-8"))
    items = doc["items"] if isinstance(doc, dict) else doc
    selected, stats = select_hq_items(
        items,
        n_target=n_target,
        zh_ratio=zh_ratio,
        s_min=cfg.s_min,
        max_source_share=cfg.max_source_share,
    )
    write_hq_dataset(
        selected,
        args.out,
        mode=args.mode,
        thresholds=config_to_dict(cfg),
        stats=stats,
    )
    print(
        f"HQ dataset n={stats['n_total']} zh={stats['n_zh']} ja={stats['n_ja']} "
        f"shortfall={stats['zh_shortfall']} → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
