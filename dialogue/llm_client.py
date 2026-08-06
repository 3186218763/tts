"""DeepSeek（OpenAI 兼容）LLM 流式客户端。"""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI


class LLMClient:
    """调用 DeepSeek API 流式生成回复。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        client: AsyncOpenAI | None = None,
        max_retries: int = 1,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_retries = max_retries

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        """流式生成回复，逐个 token 产出（None/空字符串自动跳过）。"""
        # A stream cannot be safely replayed after yielding a token because the
        # retry would duplicate already displayed text. Retry only failures that
        # happen before the first token is delivered.
        attempts = 0
        while True:
            emitted = False
            try:
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    stream=True,
                )
                async for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        emitted = True
                        yield token
                return
            except Exception:
                if emitted or attempts >= self._max_retries:
                    raise
                attempts += 1
