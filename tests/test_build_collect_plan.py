import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_collect_plan import (
    CollectPlanShortfall,
    build_collect_plan_from_cache,
    estimate_zh_clips,
)


def test_build_plan_from_search_cache(tmp_path):
    search = [
        {
            "bvid": "BV1y",
            "title": "学中文 录播完整版",
            "duration": 4000,
            "author": "小电视录播姬",
            "pubdate": 1,
            "play": 10,
        },
        {
            "bvid": "BV1z",
            "title": "【切片】反应",
            "duration": 45,
            "author": "fan",
            "pubdate": 2,
            "play": 100,
        },
    ]
    cache = tmp_path / "search_results.json"
    cache.write_text(json.dumps(search), encoding="utf-8")
    out = tmp_path / "collect_plan.json"
    doc = build_collect_plan_from_cache(
        cache,
        out,
        min_duration_by_tier={"B": 600, "A": 30, "C": 1800},
        est_zh_per_hour_b=80.0,
        est_zh_per_hour_a=200.0,
        n_target=20000,
        zh_ratio=0.60,
        allow_shortfall=True,
    )
    assert out.exists()
    assert doc["version"] == 1
    assert doc["items"][0]["tier"] == "B"
    assert any(x["tier"] == "A" for x in doc["items"])


def test_estimate_zh_clips():
    items = [
        {"tier": "B", "duration": 3600},
        {"tier": "A", "duration": 1800},
    ]
    assert estimate_zh_clips(items, 80.0, 200.0) == 180.0


def test_shortfall_exit_code(tmp_path):
    search = [
        {
            "bvid": "BV1s",
            "title": "【切片】短",
            "duration": 60,
            "author": "a",
            "pubdate": 1,
            "play": 1,
        }
    ]
    cache = tmp_path / "s.json"
    cache.write_text(json.dumps(search), encoding="utf-8")
    out = tmp_path / "p.json"
    with pytest.raises(CollectPlanShortfall):
        build_collect_plan_from_cache(
            cache,
            out,
            min_duration_by_tier={"B": 600, "A": 30, "C": 1800},
            est_zh_per_hour_b=80.0,
            est_zh_per_hour_a=200.0,
            n_target=20000,
            zh_ratio=0.60,
            allow_shortfall=False,
        )
