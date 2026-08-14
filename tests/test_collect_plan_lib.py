import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_plan_lib import (
    HARD_EXCLUDED_BVIDS,
    classify_tier,
    is_hard_excluded,
    infer_risks,
    dedupe_candidates,
    priority_score,
    build_plan_entries,
)


def test_hard_excluded_set_is_narrow():
    assert "BV1Hh9XBnEPF" in HARD_EXCLUDED_BVIDS
    assert "BV1yGzyB8EVn" in HARD_EXCLUDED_BVIDS
    assert "BV1B6SEBrEXF" not in HARD_EXCLUDED_BVIDS
    assert "BV1Ck1FBXEX3" not in HARD_EXCLUDED_BVIDS
    assert "BV1Kh4y1K7Rb" not in HARD_EXCLUDED_BVIDS


def test_watch_bvid_not_hard_excluded_gets_watch_risk():
    assert is_hard_excluded("BV1B6SEBrEXF", "看看周杰伦演唱会") is False
    assert "watch" in infer_risks("看看周杰伦演唱会")


def test_reaction_clip_is_tier_a():
    assert is_hard_excluded("BV1Kh4y1K7Rb", "【切片】【真白花音】一看到这个") is False
    assert classify_tier("【切片】【真白花音】一看到这个") == "A"


def test_tier_b_chinese_chat():
    assert classify_tier("【眞白花音】学中文 直播回放") == "B"
    assert classify_tier("白菜来啦 杂谈") == "B"


def test_tier_a_classic():
    assert classify_tier("【经典老番】白菜学中文，咬字准") == "A"


def test_tier_c_generic():
    assert classify_tier("真白花音 录播 2025-11-02") == "C"


def test_concert_hard_excluded_by_title():
    assert is_hard_excluded("BVunknown", "真白花音5周年纪念3D演唱会-录播") is True


def test_dedupe_stream_by_day_keeps_preferred_uploader():
    items = [
        {
            "bvid": "BV1a",
            "title": "2026-04-22 白菜来啦",
            "duration": 1000,
            "author": "unknown",
            "pubdate": 1,
            "play": 1,
        },
        {
            "bvid": "BV1b",
            "title": "2026-04-22 白菜来啦",
            "duration": 900,
            "author": "小电视录播姬",
            "pubdate": 1,
            "play": 1,
        },
    ]
    out = dedupe_candidates(
        items, min_duration_by_tier={"B": 600, "A": 30, "C": 1800}
    )
    assert len(out) == 1
    assert out[0]["bvid"] == "BV1b"


def test_dedupe_clips_by_bvid_not_day():
    items = [
        {
            "bvid": "BV1c",
            "title": "【切片】名场面1",
            "duration": 120,
            "author": "x",
            "pubdate": 1,
            "play": 10,
        },
        {
            "bvid": "BV1d",
            "title": "【切片】名场面2",
            "duration": 90,
            "author": "x",
            "pubdate": 1,
            "play": 10,
        },
    ]
    out = dedupe_candidates(
        items, min_duration_by_tier={"B": 600, "A": 30, "C": 1800}
    )
    assert {x["bvid"] for x in out} == {"BV1c", "BV1d"}


def test_priority_b_before_a_before_c():
    b = {
        "tier": "B",
        "duration": 1000,
        "play": 100,
        "lang_hint": "zh_bias",
        "risk": [],
    }
    a = {
        "tier": "A",
        "duration": 100,
        "play": 10000,
        "lang_hint": "zh_bias",
        "risk": [],
    }
    c = {
        "tier": "C",
        "duration": 10000,
        "play": 10000,
        "lang_hint": "mixed",
        "risk": ["watch"],
    }
    assert priority_score(b) > priority_score(a) > priority_score(c)


def test_build_plan_entries_sorted():
    raw = [
        {
            "bvid": "BV1z",
            "title": "【切片】花音",
            "duration": 60,
            "author": "a",
            "pubdate": 2,
            "play": 5,
        },
        {
            "bvid": "BV1y",
            "title": "学中文 录播",
            "duration": 3600,
            "author": "小电视录播姬",
            "pubdate": 1,
            "play": 5,
        },
    ]
    plan = build_plan_entries(raw)
    assert plan[0]["tier"] == "B"
    assert plan[0]["status"] == "planned"
    assert plan[0]["bvid"] == "BV1y"
