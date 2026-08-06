"""Deterministic text-quality gates for Huayin ASR transcripts.

These checks catch Whisper hallucinations that slip past language/logprob
filters: wrong script for the claimed language, phrase loops, Hangul
contamination, Latin-only labels on JP/ZH, and filler/elongation noise.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


# NOTE: do not use raw strings for \u escapes — r"\u3000" is not a codepoint.
_PUNCT_RE = re.compile(
    "[\t\n\r\f\v "
    "\u3000-\u303f"  # CJK punctuation (includes 〜 U+301C)
    "\uff00-\uffef"  # fullwidth forms
    r"\-_/\\|~"
    "～〜ー…·•.,!?;:\"'“”‘’、。！？【】（）()\\[\\]{}<>《》「」『』"
    "]+"
)
_REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")
_HANGUL_RE = re.compile("[\uac00-\ud7af]")
_KANA_RE = re.compile("[\u3040-\u30ff]")
_CJK_RE = re.compile("[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class TextQualityResult:
    keep: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | str]


def normalize_lang(lang: str | None) -> str:
    value = str(lang or "").strip().lower()
    if value in {"jp", "jpn", "japanese"}:
        return "ja"
    if value in {"cn", "zho", "chinese", "zh-cn", "zh-tw"}:
        return "zh"
    if value in {"en", "eng", "english"}:
        return "en"
    return value


def content_chars(text: str) -> str:
    """Strip whitespace/punctuation and collapse long elongations for content checks."""

    stripped = _PUNCT_RE.sub("", text or "")
    # collapse 3+ identical chars to one for "meaningful content" length
    return re.sub(r"(.)\1{2,}", r"\1", stripped)


def script_ratio(text: str, lang: str) -> float:
    """Fraction of non-space characters belonging to the claimed language script."""

    lang = normalize_lang(lang)
    chars = [ch for ch in (text or "") if not ch.isspace()]
    if not chars:
        return 0.0

    if lang == "ja":
        good = sum(
            1
            for ch in chars
            if _KANA_RE.fullmatch(ch)
            or _CJK_RE.fullmatch(ch)
            or ch in "ー～…、。！？"
        )
    elif lang == "zh":
        good = sum(
            1
            for ch in chars
            if _CJK_RE.fullmatch(ch) or ch in "，。！？、…；："
        )
    elif lang == "en":
        good = sum(1 for ch in chars if ch.isascii() and (ch.isalnum() or ch in "'-"))
    else:
        return 0.0
    return good / len(chars)


def latin_ratio(text: str) -> float:
    chars = [ch for ch in (text or "") if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if _LATIN_RE.fullmatch(ch)) / len(chars)


def is_char_repetitive(text: str, threshold: float = 0.5) -> bool:
    chars = [ch for ch in (text or "") if not ch.isspace()]
    if not chars:
        return False
    return max(Counter(chars).values()) / len(chars) > threshold


def has_phrase_loop(text: str, *, min_unit: int = 2, max_unit: int = 8, min_repeats: int = 3) -> bool:
    """Detect consecutive phrase loops like やばいやばいやばい or 什么什么什么什么."""

    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < min_unit * min_repeats:
        return False
    for size in range(min_unit, max_unit + 1):
        # consecutive repeats of the same unit
        pattern = re.compile(rf"(.{{{size}}})\1{{{min_repeats - 1},}}")
        if pattern.search(compact):
            return True
    return False


def has_runaway_char_repeat(text: str) -> bool:
    return bool(_REPEAT_CHAR_RE.search(text or ""))


def has_hangul(text: str) -> bool:
    return bool(_HANGUL_RE.search(text or ""))


def is_mostly_nonlexical(text: str) -> bool:
    """True when almost nothing remains after removing punctuation/elongation."""

    return len(content_chars(text)) < 2


def evaluate_text_quality(
    text: str,
    lang: str | None,
    *,
    min_script_ratio: float = 0.50,
    max_latin_ratio_cjk: float = 0.45,
    min_content_chars: int = 2,
    allow_en: bool = True,
) -> TextQualityResult:
    """Return keep/drop decision and machine-readable reasons."""

    raw = (text or "").replace("|", "/").strip()
    lang_n = normalize_lang(lang)
    reasons: list[str] = []
    metrics: dict[str, float | int | str] = {
        "lang": lang_n,
        "text_len": len(raw),
        "content_len": len(content_chars(raw)),
        "script_ratio": script_ratio(raw, lang_n),
        "latin_ratio": latin_ratio(raw),
    }

    if not raw or len(raw) < 2:
        reasons.append("short_text")
    if lang_n not in {"ja", "zh", "en"}:
        reasons.append("unsupported_lang")
    elif lang_n == "en" and not allow_en:
        reasons.append("en_disabled")

    if has_hangul(raw):
        reasons.append("hangul")
    if is_char_repetitive(raw):
        reasons.append("repetitive_char")
    if has_phrase_loop(raw):
        reasons.append("phrase_loop")
    if has_runaway_char_repeat(raw):
        reasons.append("char_run")
    if is_mostly_nonlexical(raw) or metrics["content_len"] < min_content_chars:
        reasons.append("nonlexical")

    sratio = float(metrics["script_ratio"])
    if lang_n in {"ja", "zh"} and sratio < min_script_ratio:
        reasons.append("script_mismatch")
    if lang_n == "ja" and not _KANA_RE.search(raw) and not _CJK_RE.search(raw):
        reasons.append("ja_without_jp_script")
    if lang_n == "zh" and not _CJK_RE.search(raw):
        reasons.append("zh_without_cjk")
    if lang_n in {"ja", "zh"} and float(metrics["latin_ratio"]) > max_latin_ratio_cjk:
        # allow short mixed lines if there is still enough native script
        if sratio < 0.70 or float(metrics["latin_ratio"]) > 0.60:
            reasons.append("latin_heavy")

    # pure digit / symbol spam
    alnum = [ch for ch in raw if ch.isalnum()]
    if alnum and sum(1 for ch in alnum if _DIGIT_RE.fullmatch(ch)) / len(alnum) > 0.80:
        reasons.append("digit_heavy")

    # normalize unicode weirdness (control chars)
    if any(unicodedata.category(ch)[0] == "C" and ch not in "\n\t" for ch in raw):
        reasons.append("control_chars")

    keep = not reasons
    return TextQualityResult(keep=keep, reasons=tuple(reasons), metrics=metrics)


def similarity_ratio(a: str, b: str) -> float:
    """Normalized similarity in [0, 1] via SequenceMatcher on compact text."""

    from difflib import SequenceMatcher

    left = re.sub(r"\s+", "", a or "")
    right = re.sub(r"\s+", "", b or "")
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def summarize_reasons(results: Iterable[TextQualityResult]) -> Counter:
    counter: Counter[str] = Counter()
    for result in results:
        if result.keep:
            counter["keep"] += 1
        else:
            counter["drop"] += 1
            for reason in result.reasons:
                counter[reason] += 1
    return counter
