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
    ):
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        """流式生成回复，逐个 token 产出（None/空字符串自动跳过）。"""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
