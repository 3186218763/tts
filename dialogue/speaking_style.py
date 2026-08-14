"""说话语气闭集、参考音库解析、流式标签先行解析。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SPEAKING_STYLES = frozenset({"日常", "元气", "温柔", "俏皮", "倔强", "惊讶"})
DEFAULT_SPEAKING_STYLE = "日常"

# 标签必须在台词前；解析后不得朗读、不得进 UI。
_STYLE_TAG_RE = re.compile(
    r"^[【\[]\s*说话语气\s*[:：]\s*(日常|元气|温柔|俏皮|倔强|惊讶)\s*[】\]]\s*"
)
_STYLE_LINE_RE = re.compile(
    r"^\s*说话语气\s*[:：]\s*(日常|元气|温柔|俏皮|倔强|惊讶)\s*(?:\n|$)"
)

# 缓冲超过此长度仍无完整标签 → 视为无标签，回落默认
_MAX_PREFIX_BUFFER = 48

_DEFAULT_BANK = Path(__file__).resolve().parent.parent / "model" / "refs" / "speaking_style_refs.json"


@dataclass(frozen=True)
class ReferenceClip:
    """一条 GPT-SoVITS 参考音（路径 + 转写）。"""

    style: str
    audio_path: str
    prompt_text: str
    prompt_lang: str = "zh"
    clip_id: str = ""


def normalize_speaking_style(value: str | None) -> str:
    """非法/空标签回落默认日常。"""
    if not value:
        return DEFAULT_SPEAKING_STYLE
    text = str(value).strip()
    if text in SPEAKING_STYLES:
        return text
    return DEFAULT_SPEAKING_STYLE


def strip_speaking_style_prefix(text: str) -> str:
    """去掉正文前的说话语气标签（保存历史 / 二次清洗用）。"""
    if not text:
        return text
    stripped = text.lstrip()
    match = _STYLE_TAG_RE.match(stripped)
    if match:
        return stripped[match.end() :]
    match = _STYLE_LINE_RE.match(stripped)
    if match:
        return stripped[match.end() :]
    return text


class StylePrefixParser:
    """从流式 token 中解析「标签先行」前缀，吐出可朗读正文。"""

    _TAG_STARTS = ("【", "[", "说")

    def __init__(self, default_style: str = DEFAULT_SPEAKING_STYLE):
        self._default = normalize_speaking_style(default_style)
        self._buffer = ""
        self._resolved = False
        self._style = self._default

    @property
    def resolved(self) -> bool:
        return self._resolved

    @property
    def style(self) -> str:
        return self._style if self._resolved else self._default

    def feed(self, token: str) -> str:
        if not token:
            return ""
        if self._resolved:
            return token
        self._buffer += token
        return self._try_resolve()

    def flush(self) -> str:
        if not self._resolved:
            return self._force_default()
        leftover, self._buffer = self._buffer, ""
        return leftover

    def _try_resolve(self) -> str:
        stripped = self._buffer.lstrip()

        match = _STYLE_TAG_RE.match(stripped)
        if match:
            return self._accept(match.group(1), stripped[match.end() :])

        match = _STYLE_LINE_RE.match(stripped)
        if match:
            return self._accept(match.group(1), stripped[match.end() :])

        if not stripped:
            return ""

        # 可能仍在输出标签前缀
        if any(stripped.startswith(s) or s.startswith(stripped) for s in self._TAG_STARTS):
            # 「说」也可能是正文「说什么呢」——仅当仍匹配标签前缀时继续等
            if stripped.startswith("说") and not (
                "说话语气".startswith(stripped) or stripped.startswith("说话语气")
            ):
                return self._force_default()
            if len(self._buffer) < _MAX_PREFIX_BUFFER:
                return ""
            return self._force_default()

        return self._force_default()

    def _accept(self, style: str, rest: str) -> str:
        self._style = style
        self._resolved = True
        self._buffer = ""
        return rest

    def _force_default(self) -> str:
        self._style = self._default
        self._resolved = True
        speech = self._buffer
        self._buffer = ""
        return speech


class SpeakingStyleRefBank:
    """从 speaking_style_refs.json 解析主参考音。"""

    def __init__(self, bank_path: str | Path | None = None, *, project_root: str | Path | None = None):
        self._root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
        path = Path(bank_path) if bank_path else _DEFAULT_BANK
        if not path.is_absolute():
            path = self._root / path
        self._path = path
        self._clips: dict[str, ReferenceClip] = {}
        self._default_style = DEFAULT_SPEAKING_STYLE
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self._default_style = normalize_speaking_style(data.get("default_style"))
        styles = data.get("styles") or {}
        if not isinstance(styles, dict):
            return
        for name, block in styles.items():
            style = normalize_speaking_style(name)
            if style != name:
                continue
            primary = (block or {}).get("primary") or {}
            audio = primary.get("audio")
            text = primary.get("prompt_text") or primary.get("text")
            if not audio or not text:
                continue
            audio_path = Path(str(audio))
            if not audio_path.is_absolute():
                audio_path = self._root / audio_path
            self._clips[style] = ReferenceClip(
                style=style,
                audio_path=str(audio_path.resolve()),
                prompt_text=str(text).strip(),
                prompt_lang=str(primary.get("lang") or primary.get("prompt_lang") or "zh"),
                clip_id=str(primary.get("id") or style),
            )

    def resolve(self, style: str | None) -> ReferenceClip | None:
        """返回该说话语气的主参考音；库缺失时 None（调用方用配置默认 ref）。"""
        key = normalize_speaking_style(style)
        clip = self._clips.get(key)
        if clip is not None:
            return clip
        return self._clips.get(self._default_style)

    @property
    def default_style(self) -> str:
        return self._default_style
