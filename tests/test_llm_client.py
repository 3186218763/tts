import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dialogue.llm_client import LLMClient, SUMMARY_SYSTEM_PROMPT


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


@pytest.mark.asyncio
async def test_stream_chat_retries_initial_request_once():
    calls = 0

    async def mock_create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary timeout")

        async def _stream():
            yield _mock_chunk("重试成功")

        return _stream()

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = mock_create

    client = LLMClient("fake", "fake", "test", client=mock_openai)
    tokens = [t async for t in client.stream_chat([])]

    assert tokens == ["重试成功"]
    assert calls == 2


@pytest.mark.asyncio
async def test_stream_chat_does_not_retry_after_partial_response():
    calls = 0

    async def mock_create(**kwargs):
        nonlocal calls
        calls += 1

        async def _stream():
            yield _mock_chunk("部分")
            raise RuntimeError("connection lost")

        return _stream()

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = mock_create

    client = LLMClient("fake", "fake", "test", client=mock_openai)
    with pytest.raises(RuntimeError, match="connection lost"):
        _ = [t async for t in client.stream_chat([])]

    assert calls == 1


@pytest.mark.asyncio
async def test_stream_chat_sends_natural_conversation_parameters():
    captured = {}

    async def mock_create(**kwargs):
        captured.update(kwargs)

        async def _stream():
            yield _mock_chunk("自然回复")

        return _stream()

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = mock_create
    client = LLMClient(
        "fake",
        "fake",
        "test",
        client=mock_openai,
        temperature=0.75,
        max_tokens=320,
        frequency_penalty=0.2,
    )

    _ = [token async for token in client.stream_chat([])]

    assert captured["temperature"] == 0.75
    assert captured["max_tokens"] == 320
    assert captured["frequency_penalty"] == 0.2


@pytest.mark.asyncio
async def test_summarize_chat_merges_previous_memory_and_archived_turns():
    captured = {}
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = " 用户叫小明，喜欢爵士乐。 "

    async def mock_create(**kwargs):
        captured.update(kwargs)
        return response

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = mock_create
    client = LLMClient("fake", "fake", "test", client=mock_openai)

    summary = await client.summarize_chat(
        previous_summary="用户住在上海。",
        messages=[
            {"role": "user", "content": "我叫小明，喜欢爵士乐"},
            {"role": "assistant", "content": "记住啦"},
        ],
        max_chars=100,
    )

    assert summary == "用户叫小明，喜欢爵士乐。"
    assert captured["stream"] is False
    assert captured["temperature"] == 0.2
    assert "用户住在上海" in captured["messages"][1]["content"]
    assert "我叫小明" in captured["messages"][1]["content"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"max_tokens": 0},
        {"frequency_penalty": 3},
        {"max_retries": -1},
    ],
)
def test_rejects_invalid_generation_settings(kwargs):
    with pytest.raises(ValueError):
        LLMClient("fake", "fake", "test", client=AsyncMock(), **kwargs)


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_accepts_valid_protocols(protocol):
    LLMClient("fake", "fake", "test", client=MagicMock(), protocol=protocol)


def test_rejects_invalid_protocol():
    with pytest.raises(ValueError, match="protocol"):
        LLMClient("fake", "fake", "test", client=MagicMock(), protocol="gpt")


def test_anthropic_protocol_uses_injected_http_client():
    http = MagicMock()
    client = LLMClient(
        "fake", "https://opencode.ai/zen/go", "test",
        client=http, protocol="anthropic",
    )
    assert client._client is http
