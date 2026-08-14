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


def test_requires_speaking_style_tag_protocol():
    prompt = get_system_prompt()
    assert "说话语气" in prompt
    assert "【说话语气:日常】" in prompt
    for style in ("日常", "元气", "温柔", "俏皮", "倔强", "惊讶"):
        assert f"【说话语气:{style}】" in prompt or style in prompt


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
    # 口癖密度须写成邻近区间，不能只散落出现数字 0 与 2
    assert "口癖" in prompt
    assert (
        "0-2" in prompt
        or "0～2" in prompt
        or "0–2" in prompt
        or "0-2个" in prompt
        or "0～2个" in prompt
        or "0–2个" in prompt
    )


def test_persona_contradiction_traits():
    prompt = get_system_prompt()
    for trait in ("元气", "屑", "笨", "温柔", "倔"):
        assert trait in prompt


def test_persona_forbids_graduation_as_joke():
    """毕业语气红线：不玩成笑料，且「啊？」仅用户主动提起时轻接。"""
    prompt = get_system_prompt()
    assert "玩成笑料" in prompt
    assert "啊？" in prompt
    assert "主动提起" in prompt or "用户主动" in prompt


def test_persona_t0_tightening_invariants():
    """T0 短板对应的 prompt 不变量（E01/E13/E16/E17-E18）。"""
    prompt = get_system_prompt()
    # E13 断手：禁止编造三次元故事
    assert "断手" in prompt
    assert "编造" in prompt
    # E17/E18 反列表：冒号/分号条目化禁止
    assert "冒号" in prompt or "分号" in prompt
    assert "条目" in prompt
    # E16 私事/小秘密不展开
    assert "小秘密" in prompt or "私事" in prompt
    # E01 问候/闲聊禁营业长篇
    assert "营业长篇" in prompt
