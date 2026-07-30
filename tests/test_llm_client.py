from unittest.mock import AsyncMock, MagicMock

import pytest

from dialogue.llm_client import LLMClient


def _mock_chunk(content):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    return chunk


@pytest.mark.asyncio
async def test_stream_chat_yields_tokens():
    chunks = [_mock_chunk("你好"), _mock_chunk("呀"), _mock_chunk(None), _mock_chunk("世界")]

    async def mock_create(**kwargs):
        async def _stream():
            for chunk in chunks:
                yield chunk
        return _stream()

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = mock_create

    client = LLMClient("fake", "fake", "test-model", client=mock_openai)
    tokens = [t async for t in client.stream_chat([{"role": "user", "content": "hi"}])]

    assert tokens == ["你好", "呀", "世界"]


@pytest.mark.asyncio
async def test_stream_chat_skips_empty_tokens():
    chunks = [_mock_chunk(""), _mock_chunk("内容")]

    async def mock_create(**kwargs):
        async def _stream():
            for chunk in chunks:
                yield chunk
        return _stream()

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = mock_create

    client = LLMClient("fake", "fake", "test", client=mock_openai)
    tokens = [t async for t in client.stream_chat([])]

    assert tokens == ["内容"]
