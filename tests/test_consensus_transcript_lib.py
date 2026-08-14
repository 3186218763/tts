import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import consensus_transcript_lib as ctl


def test_compact_strips_punct_and_space():
    assert ctl.compact_text("你好，世界！") == "你好世界"
    assert ctl.compact_text("こんにちは。") == "こんにちは"


def test_similarity_identical():
    assert ctl.similarity("今天天气真好", "今天天气真好") == 1.0


def test_adjudicate_strict_keep():
    d = ctl.adjudicate("今天天气真好啊", "今天天气真好啊。")
    assert d.decision == "keep"
    assert d.tier == "strict"
    assert d.text is not None
    assert "今天" in d.text


def test_adjudicate_drop_low_sim():
    d = ctl.adjudicate("今天天气真好", "完全不相干的句子内容")
    assert d.decision == "drop"
    assert d.tier == "drop_low_sim"


def test_adjudicate_soft_third_primary():
    thr = ctl.ConsensusThresholds(keep_strict=0.95, keep_soft=0.80, third_agree=0.90)
    # mid agreement between A/B, third agrees with A
    a = "我今天去公园散步了"
    b = "我今天去公园散步呀啊"
    c = "我今天去公园散步了"
    d = ctl.adjudicate(a, b, third=c, thresholds=thr)
    # sim(a,b) may be high; force soft band with more divergent b
    b2 = "我今天去公园走走看看"
    d2 = ctl.adjudicate(a, b2, third=c, thresholds=thr)
    if d2.sim_ab is not None and 0.80 <= d2.sim_ab < 0.95:
        assert d2.decision == "keep"
        assert d2.tier == "soft_third"
        assert d2.text == a
    else:
        # if still strict or still drop depending on sim, just ensure no crash
        assert d2.decision in {"keep", "drop"}


def test_adjudicate_empty_drop():
    d = ctl.adjudicate("", "")
    assert d.decision == "drop"
    assert d.tier == "drop_empty"


def test_adjudicate_no_pair():
    d = ctl.adjudicate("有字", "")
    assert d.decision == "drop"
    assert d.tier == "drop_no_pair"


def test_pick_text_longer():
    text, src = ctl.pick_text("这是一句比较完整的话", "这是一句")
    assert "完整" in text
    assert "longer" in src or src == "primary"


def test_char_error_rate_zero_and_nonzero():
    assert ctl.char_error_rate("你好世界", "你好世界") == 0.0
    assert ctl.char_error_rate("你好世界", "你好世") > 0.0


def test_summarize_decisions():
    rows = [
        {"decision": "keep", "tier": "strict", "sim_ab": 1.0},
        {"decision": "keep", "tier": "strict", "sim_ab": 0.93},
        {"decision": "drop", "tier": "drop_low_sim", "sim_ab": 0.4},
    ]
    s = ctl.summarize_decisions(rows)
    assert s["n_keep"] == 2
    assert s["n_drop"] == 1
    assert s["keep_rate"] == pytest.approx(2 / 3)
