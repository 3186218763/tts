"""Tests for speaking-style tag parsing and ref bank."""

from pathlib import Path

from dialogue.speaking_style import (
    DEFAULT_SPEAKING_STYLE,
    SpeakingStyleRefBank,
    StylePrefixParser,
    normalize_speaking_style,
    strip_speaking_style_prefix,
)


def test_normalize_falls_back_to_default():
    assert DEFAULT_SPEAKING_STYLE == "日常"
    assert normalize_speaking_style("倔强") == "倔强"
    assert normalize_speaking_style("日常") == "日常"
    assert normalize_speaking_style("低落") == DEFAULT_SPEAKING_STYLE
    assert normalize_speaking_style(None) == DEFAULT_SPEAKING_STYLE


def test_strip_prefix_variants():
    assert strip_speaking_style_prefix("【说话语气:俏皮】哼，笨蛋。") == "哼，笨蛋。"
    assert strip_speaking_style_prefix("[说话语气：温柔]谢谢尼们。") == "谢谢尼们。"
    assert strip_speaking_style_prefix("说话语气:惊讶\n诶？") == "诶？"
    assert strip_speaking_style_prefix("你好。") == "你好。"


def test_parser_tag_first_streaming():
    parser = StylePrefixParser()
    assert parser.feed("【说话语") == ""
    assert parser.resolved is False
    assert parser.feed("气:倔强】谁说我小！") == "谁说我小！"
    assert parser.resolved is True
    assert parser.style == "倔强"
    assert parser.feed("我是大大的。") == "我是大大的。"


def test_parser_no_tag_defaults_to_richang():
    parser = StylePrefixParser()
    assert parser.feed("你好呀，今天也好元气～") == "你好呀，今天也好元气～"
    assert parser.style == "日常"


def test_ref_bank_loads_primary_clips():
    bank = SpeakingStyleRefBank()
    for style in ("日常", "元气", "温柔", "俏皮", "倔强", "惊讶"):
        clip = bank.resolve(style)
        assert clip is not None, style
        assert Path(clip.audio_path).is_file(), clip.audio_path
        assert clip.prompt_text.strip()
    default = bank.resolve("不存在")
    assert default is not None
    assert default.style == "日常"
    assert "吃饭" in bank.resolve("日常").prompt_text
