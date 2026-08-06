import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dataset_text_quality as tq


@pytest.mark.parametrize(
    ("text", "lang", "expected_keep", "expected_reason"),
    [
        ("こんにちは、今日も配信だよ", "ja", True, None),
        ("大家好今天也来直播啦", "zh", True, None),
        ("やばいやばいやばいやばい", "ja", False, "phrase_loop"),
        ("什么什么什么什么白菜", "zh", False, "phrase_loop"),
        ("啊啊啊啊啊啊啊啊", "zh", False, "repetitive_char"),
        ("알아요. 가요", "zh", False, "hangul"),
        ("I love youI love you", "ja", False, "script_mismatch"),
        ("NANANANANANANANANANA", "zh", False, "script_mismatch"),
        ("ん〜〜〜〜〜〜〜〜〜", "ja", False, "nonlexical"),
        ("BGM", "ja", False, "ja_without_jp_script"),
        ("5.20M", "zh", False, "zh_without_cjk"),
        ("努力!未来!A beautiful star!", "ja", False, "latin_heavy"),
    ],
)
def test_evaluate_text_quality_cases(text, lang, expected_keep, expected_reason):
    result = tq.evaluate_text_quality(text, lang)
    assert result.keep is expected_keep
    if expected_reason is not None:
        assert expected_reason in result.reasons


def test_en_can_be_disabled():
    result = tq.evaluate_text_quality("hello there everyone", "en", allow_en=False)
    assert result.keep is False
    assert "en_disabled" in result.reasons


def test_similarity_ratio_close_and_far():
    assert tq.similarity_ratio("こんにちは", "こんにちは") == 1.0
    assert tq.similarity_ratio("大家好", "大家好呀") > 0.5
    assert tq.similarity_ratio("こんにちは", "完全不同的句子") < 0.3
