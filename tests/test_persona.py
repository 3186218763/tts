from dialogue.persona import get_system_prompt


def test_returns_non_empty_string():
    prompt = get_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_mentions_huayin_identity():
    assert "花音" in get_system_prompt()


def test_instructs_not_to_break_character():
    prompt = get_system_prompt()
    assert "AI" in prompt or "语言模型" in prompt


def test_mentions_japanese_capability():
    assert "日" in get_system_prompt()
