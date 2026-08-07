"""将 LLM 流式输出的 token 切分为完整句子。"""


class SentenceStreamer:
    """积累 token，在句子边界或长度超限时输出完整句子。

    切分规则：
    1. 遇到句末标点（。！？!?；;\\n…）且片段足够长时切出句子
    2. 缓冲区超过 max_chars 时，优先在括号外的逗号或空格处切分
    3. 没有自然边界时，允许句子增长到硬上限后再安全切分
    """

    _ENDINGS = set("。！？!?；;\n…")
    _COMMAS = set("，,、")
    _BRACKETS = {
        "(": ")",
        "（": "）",
        "[": "]",
        "【": "】",
        "{": "}",
        "｛": "｝",
        "<": ">",
        "「": "」",
        "『": "』",
    }

    def __init__(self, max_chars: int = 50, *, min_chars: int = 4):
        if min_chars < 1:
            raise ValueError("min_chars must be positive")
        if max_chars < min_chars:
            raise ValueError("max_chars must be at least min_chars")
        self._buffer = ""
        self._max_chars = max_chars
        self._min_chars = min_chars
        self._hard_max_chars = max_chars + max(10, max_chars // 2)

    def add_token(self, token: str) -> list[str]:
        """添加一个 token，返回由此产生的完整句子列表。"""
        self._buffer += token
        return self._drain()

    def flush(self) -> str | None:
        """返回缓冲区中剩余内容，无内容返回 None。"""
        if self._buffer.strip():
            result = self._buffer.strip()
            self._buffer = ""
            return result
        return None

    def _drain(self) -> list[str]:
        sentences = []
        while True:
            extracted = self._try_extract()
            if extracted is None:
                break
            if extracted.strip():
                sentences.append(extracted)
        return sentences

    def _try_extract(self) -> str | None:
        bracket_stack: list[str] = []
        in_asterisks = False
        natural_splits: list[int] = []
        safe_splits: list[int] = []

        # Track boundaries outside brackets so neither natural nor hard
        # splitting exposes half of a stage direction to the speech filter.
        for i, ch in enumerate(self._buffer):
            if ch == "*" and (i == 0 or self._buffer[i - 1] != "\\"):
                in_asterisks = not in_asterisks
                if not in_asterisks and not bracket_stack:
                    safe_splits.append(i + 1)
                continue
            if ch in self._BRACKETS:
                bracket_stack.append(self._BRACKETS[ch])
                continue
            if bracket_stack and ch == bracket_stack[-1]:
                bracket_stack.pop()
                if not bracket_stack and not in_asterisks:
                    safe_splits.append(i + 1)
                continue

            if bracket_stack or in_asterisks:
                continue

            safe_splits.append(i + 1)
            if ch in self._ENDINGS:
                sentence = self._buffer[: i + 1]
                visible_chars = len("".join(sentence.split()))
                if visible_chars >= self._min_chars:
                    self._buffer = self._buffer[i + 1 :]
                    return sentence
            elif ch in self._COMMAS or ch.isspace():
                natural_splits.append(i + 1)

        # A split position is a one-based offset, so a boundary exactly at
        # min_chars is valid.
        if len(self._buffer) > self._max_chars:
            search_end = min(len(self._buffer), self._hard_max_chars)
            for split_at in reversed(natural_splits):
                if split_at <= search_end and len(
                    "".join(self._buffer[:split_at].split())
                ) >= self._min_chars:
                    sentence = self._buffer[:split_at]
                    self._buffer = self._buffer[split_at:]
                    return sentence

            # When the nominal hard position is inside protected text, choose
            # the nearest safe position instead of cutting the bracketed span.
            if len(self._buffer) > self._hard_max_chars:
                before_limit = [
                    position
                    for position in safe_splits
                    if position <= self._hard_max_chars
                ]
                after_limit = [
                    position
                    for position in safe_splits
                    if position > self._hard_max_chars
                ]
                split_at = (
                    before_limit[-1]
                    if before_limit
                    else (after_limit[0] if after_limit else None)
                )
                if split_at is not None:
                    sentence = self._buffer[:split_at]
                    self._buffer = self._buffer[split_at:]
                    return sentence

        return None
