"""Build Chinese-first collect plan from search cache or live search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_plan_lib import build_plan_entries  # noqa: E402
from dataset_hq_config import load_dataset_hq_config  # noqa: E402

DEFAULT_KEYWORDS = [
    "真白花音 学中文",
    "眞白花音 学中文",
    "真白花音 中文",
    "真白花音 杂谈 录播",
    "眞白花音 白菜来啦",
    "真白花音 切片",
    "眞白花音 切片",
    "真白花音 经典老番",
    "真白花音 录播",
    "眞白花音 录播",
    "真白花音 直播回放",
]


class CollectPlanShortfall(RuntimeError):
    """Raised when estimated ZH clips are below target and shortfall not allowed."""


def estimate_zh_clips(
    items: list[Mapping[str, Any]],
    per_b: float,
    per_a: float,
) -> float:
    total = 0.0
    for item in items:
        hours = float(item.get("duration") or 0) / 3600.0
        tier = str(item.get("tier") or "")
        if tier == "B":
            total += hours * per_b
        elif tier == "A":
            total += hours * per_a
    return total


def build_collect_plan_from_cache(
    cache_path: Path,
    out_path: Path,
    *,
    min_duration_by_tier: Mapping[str, int],
    est_zh_per_hour_b: float,
    est_zh_per_hour_a: float,
    n_target: int,
    zh_ratio: float,
    allow_shortfall: bool,
) -> dict[str, Any]:
    raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        search = list(raw["items"])
    else:
        search = list(raw)

    items = build_plan_entries(search, min_duration_by_tier=min_duration_by_tier)
    est = estimate_zh_clips(items, est_zh_per_hour_b, est_zh_per_hour_a)
    need = zh_ratio * n_target
    shortfall = est < need
    if shortfall and not allow_shortfall:
        raise CollectPlanShortfall(
            f"estimated ZH clips {est:.1f} < need {need:.1f} "
            f"(zh_ratio={zh_ratio} * n_target={n_target})"
        )

    doc = {
        "version": 1,
        "estimated_zh_clips": est,
        "estimated_zh_shortfall": shortfall,
        "items": items,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Chinese-first collect plan")
    p.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="Load search results JSON instead of live search",
    )
    p.add_argument(
        "--search",
        action="store_true",
        help="Search Bilibili with default keywords (writes search cache)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "collect_plan.json",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dataset_hq.yaml",
    )
    p.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Write plan even when estimated ZH clips are below target",
    )
    p.add_argument("--max-pages", type=int, default=5)
    args = p.parse_args(argv)

    cfg = load_dataset_hq_config(args.config)
    cache = args.from_cache
    if args.search:
        import batch_download as bd

        client = bd.init_client()
        all_results: dict[str, dict] = {}
        for kw in DEFAULT_KEYWORDS:
            print(f"search: {kw}")
            for r in bd.search_videos(client, kw, max_pages=args.max_pages):
                all_results[r["bvid"]] = r
        cache = PROJECT_ROOT / "data" / "search_results.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(list(all_results.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"search cache: {cache} ({len(all_results)} items)")

    if cache is None:
        cache = PROJECT_ROOT / "data" / "search_results.json"
    if not cache.exists():
        print(f"search cache missing: {cache}", file=sys.stderr)
        return 1

    try:
        doc = build_collect_plan_from_cache(
            cache,
            args.output,
            min_duration_by_tier=cfg.min_duration_by_tier,
            est_zh_per_hour_b=cfg.est_zh_clips_per_hour_b,
            est_zh_per_hour_a=cfg.est_zh_clips_per_hour_a,
            n_target=cfg.n_target,
            zh_ratio=cfg.zh_ratio,
            allow_shortfall=args.allow_shortfall,
        )
    except CollectPlanShortfall as exc:
        print(f"shortfall: {exc}", file=sys.stderr)
        return 2

    print(
        f"plan items={len(doc['items'])} "
        f"est_zh={doc['estimated_zh_clips']:.1f} "
        f"shortfall={doc['estimated_zh_shortfall']} → {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
