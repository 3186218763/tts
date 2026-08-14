import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hq_dataset_lib import select_hq_items, normalize_text_key


def _item(path, lang, score, text, bvid="BVx", decision="candidate"):
    return {
        "path": path,
        "lang": lang,
        "score": score,
        "text": text,
        "source_bvid": bvid,
        "decision": decision,
        "tier": "B",
    }


def test_normalize_text_key_strips_punct():
    assert normalize_text_key("你好！！") == normalize_text_key("你好")


def test_zh_shortfall_no_jp_padding():
    items = [
        _item(f"/zh{i}.wav", "zh", 90 - i * 0.01, f"中文句子{i}", "BVz")
        for i in range(100)
    ]
    items += [
        _item(f"/ja{i}.wav", "ja", 99, f"にほんご{i}", "BVj") for i in range(500)
    ]
    selected, stats = select_hq_items(
        items, n_target=200, zh_ratio=0.60, s_min=55.0, max_source_share=1.0
    )
    assert stats["zh_shortfall"] is True
    assert stats["n_total"] == 166
    assert stats["n_zh"] == 100
    assert stats["n_ja"] == 66
    assert stats["n_zh"] / stats["n_total"] >= 0.60 - 1e-9


def test_zh_sufficient_targets_ratio():
    items = [
        _item(f"/zh{i}.wav", "zh", 90, f"中文{i}", f"BVz{i % 20}") for i in range(200)
    ]
    items += [
        _item(f"/ja{i}.wav", "ja", 90, f"日本語{i}", f"BVj{i % 20}") for i in range(200)
    ]
    selected, stats = select_hq_items(
        items, n_target=200, zh_ratio=0.60, s_min=55.0, max_source_share=1.0
    )
    assert stats["zh_shortfall"] is False
    assert stats["n_total"] == 200
    assert stats["n_zh"] == 120
    assert stats["n_ja"] == 80


def test_prefers_higher_score_within_lang():
    items = [
        _item("/a.wav", "zh", 60, "句子甲", "B1"),
        _item("/b.wav", "zh", 90, "句子乙", "B2"),
    ]
    selected, stats = select_hq_items(
        items, n_target=1, zh_ratio=0.60, s_min=55.0, max_source_share=1.0
    )
    assert selected[0]["path"] == "/b.wav"


def test_dedupe_identical_text_keeps_best_score():
    items = [
        _item("/a.wav", "zh", 70, "同一句", "B1"),
        _item("/b.wav", "zh", 90, "同一句", "B2"),
    ]
    selected, stats = select_hq_items(
        items, n_target=10, zh_ratio=0.60, s_min=55.0, max_source_share=1.0
    )
    paths = {x["path"] for x in selected}
    assert "/b.wav" in paths
    assert "/a.wav" not in paths


def test_source_cap():
    items = [
        _item(f"/z{i}.wav", "zh", 90, f"句{i}", "BVonly") for i in range(100)
    ]
    items += [
        _item(f"/j{i}.wav", "ja", 90, f"日{i}", f"BVj{i}") for i in range(100)
    ]
    selected, stats = select_hq_items(
        items, n_target=100, zh_ratio=0.60, s_min=55.0, max_source_share=0.15
    )
    # Cap is computed from planned n_zh = ceil(0.6 * 100) = 60 → floor(0.15*60)=9
    from_only = sum(1 for x in selected if x["source_bvid"] == "BVonly")
    assert from_only <= max(1, int(0.15 * 60))
    assert from_only <= 9


def test_stable_sort_tie_break_by_path():
    items = [
        _item("/b.wav", "zh", 80, "甲", "B1"),
        _item("/a.wav", "zh", 80, "乙", "B2"),
    ]
    selected, stats = select_hq_items(
        items, n_target=1, zh_ratio=0.6, s_min=55.0, max_source_share=1.0
    )
    assert selected[0]["path"] == "/a.wav"


def test_borderline_excluded():
    items = [_item("/a.wav", "zh", 90, "好句", "B1", decision="borderline")]
    selected, stats = select_hq_items(
        items, n_target=10, zh_ratio=0.6, s_min=55.0, max_source_share=1.0
    )
    assert selected == []
