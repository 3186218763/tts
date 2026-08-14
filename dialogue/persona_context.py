"""每轮人格上下文：事实召回（关键词 RAG）+ few-shot 台词抽取。

三层组合的第二层（事实 RAG）与第三层（few-shot）。资产位于 docs/persona/，
任何加载失败都静默降级为空列表，不阻塞对话。
"""

from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

_PERSONA_DIR = Path(__file__).resolve().parent.parent / "docs" / "persona"
_FACTS_PATH = _PERSONA_DIR / "facts-kb.json"
_FEWSHOT_PATH = _PERSONA_DIR / "fewshot-lines.json"

# 统一场景词表：以 _SCENE_KEYWORDS 的键为准（问候/感谢/自嘲/游戏/食物/唱歌/
# 情绪/惊讶/屑/告别/夸赞/被说小/被开盒），资产中全部 scenes 标签均来自该词表。
_HAS_KANA = re.compile("[\u3040-\u30ff]")
_SENSITIVE_CAT = "sensitive"
_MAX_FACTS = 4
_MAX_EXAMPLES = 5

_SCENE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "问候": ("你好", "早安", "晚安", "早上好", "晚上好", "欢迎", "在吗", "辛苦了", "こんにちは", "おはよう"),
    "感谢": ("谢谢", "感谢", "阿里嘎多", "多谢", "ありがとう"),
    "自嘲": ("菜", "笨", "不行", "没办法", "输了", "坑", "笨蛋"),
    "游戏": ("游戏", "lol", "apex", "宝可梦", "赛马娘", "雀魂", "打一把", "开黑", "段位", "ゲーム"),
    "食物": ("吃", "好吃", "麦当劳", "蜜雪", "冰淇淋", "甜", "肉", "饭", "野菜", "食べ"),
    "唱歌": ("歌", "唱", "live", "演唱会", "歌う"),
    "情绪": ("开心", "难过", "害怕", "生气", "累", "心情", "嬉しい", "悲しい"),
    "惊讶": ("什么", "真的吗", "不会吧", "震惊", "えっ"),
    "屑": ("屑", "坏", "欺负", "恶作剧"),
    "告别": ("再见", "拜拜", "毕业", "想她", "回来", "さようなら", "バイバイ"),
    "夸赞": ("可爱", "厉害", "好听", "棒", "喜欢", "すごい", "かわいい"),
    "被说小": ("小", "矮", "萝莉", "ちっちゃい"),
    "被开盒": ("开盒", "线下", "地址", "黑粉"),
}

_AVOID_HINT = (
    "<persona_facts>\n"
    "触发三次元敏感话题（男朋友/绯闻/比心等）。这是她的私事，"
    "用\"这是她的私事\"回应并自然转移话题，不要展开任何细节。\n"
    "</persona_facts>"
)


def _normalize(text: str) -> str:
    """归一化：NFKC 全角转半角、小写、去全部空白。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "").lower())


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        logger.warning("persona 资产加载失败，已降级为空上下文: %s", path)
        return None
    if not isinstance(data, dict):
        logger.warning("persona 资产格式异常（顶层非对象），已降级为空上下文: %s", path)
        return None
    return data


def _retrieve_facts(user_text: str, facts: list[dict]) -> tuple[list[dict], bool]:
    """双向子串匹配召回；返回（命中事实，是否触发敏感回避）。

    敏感类命中时不返回任何事实内容，只让调用方注入回避提示。
    """
    user = _normalize(user_text)
    if not user or not facts:
        return [], False
    hits: list[tuple[int, dict]] = []
    sensitive_hit = False
    for fact in facts:
        count = 0
        for phrase in fact.get("q", []):
            normalized = _normalize(phrase)
            if normalized and (normalized in user or user in normalized):
                count += 1
        if count:
            hits.append((count, fact))
            if fact.get("cat") == _SENSITIVE_CAT:
                sensitive_hit = True
    if sensitive_hit:
        return [], True
    hits.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in hits[:_MAX_FACTS]], False


def _matched_scenes(user_text: str) -> set[str]:
    user = _normalize(user_text)
    matched: set[str] = set()
    for scene, keywords in _SCENE_KEYWORDS.items():
        if any(_normalize(keyword) in user for keyword in keywords):
            matched.add(scene)
    return matched


def _asr_item_allowed(item: dict, matched_scenes: set[str]) -> bool:
    """late 仅告别场景可注入；无 era_tag 视为兼容旧数据可注入。"""
    if item.get("era_tag") != "late":
        return True
    return "告别" in matched_scenes


def _pick_fewshot(user_text: str, data: dict, rng: random.Random) -> list[str]:
    """按场景抽取 3-5 条台词：官方优先、中文为主、整句日语仅用户含日语时注入。"""
    official = data.get("official_comments", [])
    fan = data.get("fan_pool", [])
    pool = data.get("asr_pool", []) if data.get("asr_pool_enabled") else []
    user_has_jp = bool(_HAS_KANA.search(user_text or ""))
    chosen: list[str] = []

    def add(candidates: list[dict]) -> None:
        for item in candidates:
            if len(chosen) >= _MAX_EXAMPLES:
                return
            text = item.get("text", "").strip()
            if not text or text in chosen:
                continue
            if _HAS_KANA.search(text) and not user_has_jp:
                continue
            chosen.append(text)

    scenes = _matched_scenes(user_text)
    # late 过滤：仅对 asr_pool；无 era_tag 兼容旧数据
    pool = [item for item in pool if _asr_item_allowed(item, scenes)]
    if scenes:
        add([item for item in official + fan if scenes & set(item.get("scenes", []))])
        if pool:
            zh = [item for item in pool if "zh" == item.get("lang")]
            add([item for item in zh if scenes & set(item.get("scenes", []))])
            if user_has_jp:
                ja = [item for item in pool if "ja" == item.get("lang")]
                add([item for item in ja if scenes & set(item.get("scenes", []))])
    else:
        add(official + fan)
    if pool and len(chosen) < 3:
        zh = [item for item in pool if "zh" == item.get("lang")]
        rng.shuffle(zh)
        add(zh)
        if user_has_jp:
            ja = [item for item in pool if "ja" == item.get("lang")]
            rng.shuffle(ja)
            add(ja)
    return chosen[:_MAX_EXAMPLES]


def build_persona_context(
    user_text: str | None, *, seed: int | None = None
) -> list[dict[str, str]]:
    """构建每轮注入的 system 消息（事实 + few-shot）；任何失败返回空列表。"""
    if not user_text or not user_text.strip():
        return []
    try:
        facts_data = _load_json(_FACTS_PATH)
        fewshot_data = _load_json(_FEWSHOT_PATH)
        if facts_data is None or fewshot_data is None:
            return []
        messages: list[dict[str, str]] = []

        facts = facts_data.get("facts", [])
        if facts:
            hits, sensitive = _retrieve_facts(user_text, facts)
            if sensitive:
                # 敏感话题只触发回避提示：不注入事实、也不注入示例台词
                messages.append({"role": "system", "content": _AVOID_HINT})
                return messages
            if hits:
                body = "\n".join(f"- {fact['id']}：{fact['fact']}" for fact in hits)
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "<persona_facts>\n背景资料（仅作事实参考，不要复述）：\n"
                            f"{body}\n</persona_facts>"
                        ),
                    }
                )

        rng = random.Random(seed) if seed is not None else random.Random()
        examples = _pick_fewshot(user_text, fewshot_data, rng)
        if examples:
            body = "\n".join(f"- {text}" for text in examples)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "<persona_examples>\n花音的真实台词（模仿其语气，不要逐字复述）：\n"
                        f"{body}\n</persona_examples>"
                    ),
                }
            )
        return messages
    except Exception:
        logger.warning("persona 上下文构建失败，已降级为空上下文")
        return []
