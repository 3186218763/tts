"""Normalize LLM output before it is displayed or sent to TTS."""

from __future__ import annotations

import re
import unicodedata

from .speaking_style import strip_speaking_style_prefix


def strip_style_for_history(text: str) -> str:
    """Remove leading speaking-style control tags before history save."""
    return strip_speaking_style_prefix(text)


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_HTML_BLOCK_RE = re.compile(r"<([A-Za-z][\w:-]*)\b[^>]*>[\s\S]*?</\1>")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,200}>")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^\)\n]*\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\([^\)\n]*\)")
_BOLD_RE = re.compile(r"(?:\*\*|__)([^\n]+?)(?:\*\*|__)")
_ASTERISK_SPAN_RE = re.compile(r"(?<!\\)\*([^*\n]{1,160})\*")
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:真白花音|眞白花音|花音|ましろ[・･]?はな)\s*[：:]\s*",
    re.IGNORECASE,
)
_LEADING_PUNCTUATION_RE = re.compile(r"^[\s，。！？!?；;、,:：…~～]+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([，。！？!?；;、,:：…~～])")
_SPACE_AFTER_CJK_PUNCTUATION_RE = re.compile(
    r"([，。！？；、：…～]) +(?=[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff])"
)
_CJK_SPACE_RE = re.compile(
    r"(?<=[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]) +"
    r"(?=[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff])"
)
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_BRACKET_PATTERNS = (
    re.compile(r"（([^（）\n]{0,160})）"),
    re.compile(r"\(([^()\n]{0,160})\)"),
    re.compile(r"【([^【】\n]{0,160})】"),
    re.compile(r"\[([^\[\]\n]{0,160})\]"),
    re.compile(r"｛([^｛｝\n]{0,160})｝"),
    re.compile(r"\{([^{}\n]{0,160})\}"),
)

_ACTION_CUES = (
    "心想",
    "心里",
    "心裏",
    "心理",
    "内心",
    "內心",
    "独白",
    "獨白",
    "旁白",
    "动作",
    "動作",
    "表情",
    "语气",
    "語氣",
    "脸红",
    "臉紅",
    "害羞",
    "羞涩",
    "羞澀",
    "微笑",
    "轻笑",
    "輕笑",
    "苦笑",
    "大笑",
    "偷笑",
    "笑着",
    "笑著",
    "叹气",
    "嘆氣",
    "叹息",
    "嘆息",
    "点头",
    "點頭",
    "摇头",
    "搖頭",
    "眨眼",
    "挥手",
    "揮手",
    "鞠躬",
    "沉默",
    "小声",
    "小聲",
    "低声",
    "低聲",
    "轻声",
    "輕聲",
    "大声",
    "大聲",
    "悄悄",
    "尴尬",
    "尷尬",
    "惊讶",
    "驚訝",
    "疑惑",
    "无奈",
    "無奈",
    "开心",
    "開心",
    "生气",
    "生氣",
    "哭泣",
    "抽泣",
    "看向",
    "看着",
    "看著",
    "望向",
    "转身",
    "轉身",
    "耸肩",
    "聳肩",
    "捂脸",
    "捂臉",
    "鼓掌",
    "拍手",
    "深呼吸",
    "吸气",
    "吸氣",
    "呼气",
    "呼氣",
    "咳嗽",
    "喘气",
    "喘氣",
    "thinking",
    "thought",
    "blush",
    "smile",
    "laugh",
    "nod",
    "shake",
    "whisper",
    "sigh",
)


def _is_action(content: str) -> bool:
    normalized = content.strip().lower()
    if not normalized:
        return True
    if normalized in {"os", "旁白", "动作", "動作", "表情"}:
        return True
    return any(cue in normalized for cue in _ACTION_CUES)


def _remove_bracketed_actions(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        return " " if _is_action(content) else content

    # A few passes support simple nested bracket pairs without a greedy regex.
    for _ in range(3):
        previous = text
        for pattern in _BRACKET_PATTERNS:
            text = pattern.sub(replace, text)
        if text == previous:
            break
    return text


def _remove_unclosed_actions(text: str) -> str:
    for opener in ("（", "(", "【", "[", "｛", "{"):
        start = text.rfind(opener)
        if start >= 0 and _is_action(text[start + 1 :]):
            text = text[:start]
    if text.count("*") % 2:
        start = text.rfind("*")
        if _is_action(text[start + 1 :]):
            text = text[:start]
    return text


def _remove_emoji(text: str) -> str:
    result = []
    for char in text:
        codepoint = ord(char)
        if (
            0x1F1E6 <= codepoint <= 0x1FAFF
            or 0x2600 <= codepoint <= 0x27FF
            or codepoint in {0x200D, 0xFE0E, 0xFE0F}
            or unicodedata.category(char) == "So"
        ):
            continue
        result.append(char)
    return "".join(result)


def _has_speakable_content(text: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def normalize_speech_text(text: str) -> str | None:
    """Return dialogue-only text, or ``None`` when nothing should be spoken.

    The filter deliberately removes only bracketed spans that look like stage
    directions. Ordinary parenthetical content such as ``(2026)`` is kept.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    normalized = strip_speaking_style_prefix(text)
    normalized = _CODE_BLOCK_RE.sub(" ", normalized)
    normalized = _HTML_BLOCK_RE.sub(" ", normalized)
    normalized = _HTML_TAG_RE.sub(" ", normalized)
    normalized = _MARKDOWN_IMAGE_RE.sub(" ", normalized)
    normalized = _MARKDOWN_LINK_RE.sub(r"\1", normalized)
    normalized = _BOLD_RE.sub(r"\1", normalized)

    def replace_asterisk_span(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        return " " if _is_action(content) else content

    normalized = _ASTERISK_SPAN_RE.sub(replace_asterisk_span, normalized)
    normalized = _remove_bracketed_actions(normalized)
    normalized = _remove_unclosed_actions(normalized)
    normalized = _remove_emoji(normalized)
    normalized = _LEADING_LABEL_RE.sub("", normalized)
    normalized = re.sub(r"^[>#]+\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = _LEADING_PUNCTUATION_RE.sub("", normalized)
    normalized = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", normalized)
    normalized = _SPACE_AFTER_CJK_PUNCTUATION_RE.sub(r"\1", normalized)
    normalized = _CJK_SPACE_RE.sub("", normalized).strip()

    if not normalized or not _has_speakable_content(normalized):
        return None
    return normalized


def resolve_text_language(text: str, configured_language: str = "auto") -> str:
    """Resolve unambiguous single-language text and preserve mixed auto mode."""
    if configured_language != "auto":
        return configured_language
    has_kana = bool(_KANA_RE.search(text))
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_kana and not has_latin:
        return "ja"
    if has_cjk and not has_kana and not has_latin:
        return "zh"
    if has_latin and not has_kana and not has_cjk:
        return "en"
    return "auto"
