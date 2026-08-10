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


def test_requires_spoken_dialogue_only():
    prompt = get_system_prompt()
    assert "只输出" in prompt
    assert "心理活动" in prompt


def test_persona_card_honesty_clause_and_red_lines():
    prompt = get_system_prompt()
    assert "AI 复刻" in prompt
    assert "2026-05-01" in prompt
    assert "不制作其周边" in prompt


def test_persona_card_chinese_first_and_anti_ai_rules():
    prompt = get_system_prompt()
    assert "中文为主" in prompt
    assert "首先" in prompt and "语言模型" in prompt  # 反 AI 约束以"禁止..."形式出现
    assert "玩成笑料" in prompt  # 语气红线


def test_persona_locks_peak_era():
    prompt = get_system_prompt()
    assert "2020" in prompt and "2023" in prompt


def test_persona_tone_fingerprint_rules():
    prompt = get_system_prompt()
    assert "1-3" in prompt or "1～3" in prompt
    assert "口癖" in prompt
    assert "0" in prompt and "2" in prompt  # 口癖 0-2


def test_persona_contradiction_traits():
    prompt = get_system_prompt()
    for trait in ("元气", "屑", "笨", "温柔", "倔"):
        assert trait in prompt


def test_persona_forbids_graduation_as_joke():
    prompt = get_system_prompt()
    assert "笑料" in prompt or "玩成笑料" in prompt
