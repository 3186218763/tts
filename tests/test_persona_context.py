import json
import random
import re
from pathlib import Path

from dialogue.persona import get_system_prompt
from dialogue.persona_context import (
    _pick_fewshot,
    _retrieve_facts,
    build_persona_context,
)

ROOT = Path(__file__).resolve().parent.parent
KANA = re.compile("[\u3040-\u30ff]")


def _load_facts():
    path = ROOT / "docs/persona/facts-kb.json"
    return json.loads(path.read_text(encoding="utf-8"))["facts"]


def _synthetic_lines() -> dict:
    return {
        "asr_pool_enabled": False,
        "official_comments": [
            {"text": "阿里嘎多！红豆泥阿里嘎多", "lang": "zh", "scenes": ["感谢"]},
            {"text": "谁说我小！？我是大大的", "lang": "zh", "scenes": ["被说小"]},
        ],
        "fan_pool": [{"text": "尼们现在是什么心情", "lang": "zh", "scenes": ["告别"]}],
        "asr_pool": [],
    }


def test_fact_retrieval_grad_hits_f012():
    hits, sensitive = _retrieve_facts("花音为什么毕业", _load_facts())
    assert any(fact["id"] == "F012" for fact in hits)
    assert sensitive is False


def test_sensitive_topic_returns_no_facts():
    hits, sensitive = _retrieve_facts("花音有男朋友吗", _load_facts())
    assert hits == []
    assert sensitive is True


def test_build_context_grad_injects_f012():
    ctx = build_persona_context("为什么毕业", seed=1)
    facts_msg = next(m for m in ctx if "<persona_facts>" in m["content"])
    assert "F012" in facts_msg["content"]
    assert "永久停止" in facts_msg["content"]


def test_build_context_sensitive_only_avoid_hint():
    ctx = build_persona_context("花音有男朋友吗", seed=1)
    assert len(ctx) == 1
    assert "私事" in ctx[0]["content"]
    assert "<persona_facts>" in ctx[0]["content"]
    assert "F0" not in ctx[0]["content"]


def test_empty_or_blank_user_returns_empty():
    assert build_persona_context("") == []
    assert build_persona_context(None) == []
    assert build_persona_context("   ") == []


def test_missing_assets_returns_empty(monkeypatch):
    monkeypatch.setattr("dialogue.persona_context._FACTS_PATH", ROOT / "docs/nope.json")
    monkeypatch.setattr("dialogue.persona_context._FEWSHOT_PATH", ROOT / "docs/nope2.json")
    assert build_persona_context("你好", seed=1) == []


def test_fewshot_scene_hit_official_first():
    picked = _pick_fewshot("阿里嘎多谢谢尼们", _synthetic_lines(), random.Random(1))
    assert picked == ["阿里嘎多！红豆泥阿里嘎多"]


def test_fewshot_cap_and_zh_only_for_zh_user():
    data = _synthetic_lines()
    data["asr_pool_enabled"] = True
    data["asr_pool"] = [
        {"text": "おはようございます", "lang": "ja", "scenes": ["问候"]},
        {"text": "谢谢你呀", "lang": "zh", "scenes": ["感谢"]},
        {"text": "玩游戏好开心", "lang": "zh", "scenes": ["游戏"]},
        {"text": "晚饭吃麦当劳", "lang": "zh", "scenes": ["食物"]},
        {"text": "今天也辛苦了", "lang": "zh", "scenes": ["问候"]},
        {"text": "気持ちいいね", "lang": "ja", "scenes": ["情绪"]},
    ]
    picked = _pick_fewshot("谢谢", data, random.Random(1))
    assert 1 <= len(picked) <= 5
    assert picked[0] == "阿里嘎多！红豆泥阿里嘎多"
    assert all(not KANA.search(line) for line in picked)


def test_fewshot_jp_allowed_when_user_uses_jp():
    data = _synthetic_lines()
    data["asr_pool_enabled"] = True
    data["asr_pool"] = [{"text": "おはようございます", "lang": "ja", "scenes": ["问候"]}]
    picked = _pick_fewshot("おはよう", data, random.Random(1))
    assert any(KANA.search(line) for line in picked)


def test_fewshot_deterministic_with_seed():
    data = _synthetic_lines()
    data["asr_pool_enabled"] = True
    data["asr_pool"] = [
        {"text": f"台词{i}", "lang": "zh", "scenes": []} for i in range(10)
    ]
    first = _pick_fewshot("今天心情不错", data, random.Random(42))
    second = _pick_fewshot("今天心情不错", data, random.Random(42))
    assert first == second
    assert 1 <= len(first) <= 5


def test_persona_card_honesty_and_chinese_first():
    prompt = get_system_prompt()
    assert "AI 复刻" in prompt
    assert "中文为主" in prompt
    assert "不制作其周边" in prompt
