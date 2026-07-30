"""将 LLM 流式输出的 token 切分为完整句子。"""


class SentenceStreamer:
    """积累 token，在句子边界或长度超限时输出完整句子。

    切分规则：
    1. 遇到句末标点（。！？!?；;\\n…）立即切出句子
    2. 缓冲区超过 max_chars 且无句末标点时，在最近的逗号处切分
    3. 无逗号则在 max_chars 处强制切分
    """

    _ENDINGS = set("。！？!?；;\n…")
    _COMMAS = set("，,、")

    def __init__(self, max_chars: int = 25):
        self._buffer = ""
        self._max_chars = max_chars

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
        # 1. 句末标点切分
        for i, ch in enumerate(self._buffer):
            if ch in self._ENDINGS:
                sentence = self._buffer[: i + 1]
                self._buffer = self._buffer[i + 1 :]
                return sentence

        # 2. 超长时在逗号处切分
        if len(self._buffer) > self._max_chars:
            search_end = min(len(self._buffer), self._max_chars)
            for i in range(search_end - 1, -1, -1):
                if self._buffer[i] in self._COMMAS:
                    sentence = self._buffer[: i + 1]
                    self._buffer = self._buffer[i + 1 :]
                    return sentence
            # 3. 无逗号，强制切分
            sentence = self._buffer[: self._max_chars]
            self._buffer = self._buffer[self._max_chars :]
            return sentence

        return None
