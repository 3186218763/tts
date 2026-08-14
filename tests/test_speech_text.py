from dialogue.speech_text import normalize_speech_text


def test_removes_common_stage_directions_and_emoji():
    text = "（脸红）*轻轻点头*你好呀！💕"
    assert normalize_speech_text(text) == "你好呀！"


def test_strips_speaking_style_control_tag():
    assert normalize_speech_text("【说话语气:温柔】谢谢尼们。") == "谢谢尼们。"


def test_removes_inner_monologue_but_keeps_spoken_text():
    text = "你好。（心里想着：今天也见到你了。）真的很开心哦！"
    assert normalize_speech_text(text) == "你好。真的很开心哦！"


def test_keeps_non_action_parenthetical_content():
    assert normalize_speech_text("新版本（2026）发布啦。") == "新版本2026发布啦。"


def test_cleans_markdown_labels_and_links():
    text = "花音：**欢迎**来[我的主页](https://example.com)！"
    assert normalize_speech_text(text) == "欢迎来我的主页！"


def test_returns_none_for_action_or_symbols_only():
    assert normalize_speech_text("（害羞地低下头）💕……") is None


def test_removes_unclosed_stage_direction():
    assert normalize_speech_text("你好。*轻轻点头") == "你好。"
