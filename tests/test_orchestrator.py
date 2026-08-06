from unittest.mock import AsyncMock, MagicMock

import pytest

from dialogue.conversation import Conversation
from dialogue.orchestrator import Orchestrator


def _async_iter(tokens):
    async def _gen():
        for t in tokens:
            yield t
    return _gen()


@pytest.mark.asyncio
async def test_chat_yields_sentences_and_synthesizes_audio():
    mock_llm = AsyncMock()
    mock_llm.stream_chat = MagicMock(return_value=_async_iter(
        ["你好", "呀。", "我是", "花音。"]
    ))
    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(return_value=b"audio")
    mock_player = AsyncMock()
    mock_player.play_wav_bytes = AsyncMock()

    orch = Orchestrator(mock_llm, mock_tts, mock_player, max_chars=25)
    conv = Conversation(max_turns=10)

    sentences = [s async for s in orch.chat("hi", conv)]

    assert sentences == ["你好呀。", "我是花音。"]
    assert mock_tts.synthesize.call_count == 2
    assert mock_player.play_wav_bytes.call_count == 2


@pytest.mark.asyncio
async def test_chat_saves_conversation_history():
    mock_llm = AsyncMock()
    mock_llm.stream_chat = MagicMock(return_value=_async_iter(["你好。"]))
    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(return_value=b"audio")
    mock_player = AsyncMock()
    mock_player.play_wav_bytes = AsyncMock()

    orch = Orchestrator(mock_llm, mock_tts, mock_player)
    conv = Conversation()

    async for _ in orch.chat("hi", conv):
        pass

    messages = conv.get_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1] == {"role": "assistant", "content": "你好。"}


@pytest.mark.asyncio
async def test_chat_flushes_partial_sentence():
    mock_llm = AsyncMock()
    mock_llm.stream_chat = MagicMock(return_value=_async_iter(["你好", "呀"]))
    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(return_value=b"audio")
    mock_player = AsyncMock()
    mock_player.play_wav_bytes = AsyncMock()

    orch = Orchestrator(mock_llm, mock_tts, mock_player)
    conv = Conversation()

    sentences = [s async for s in orch.chat("hi", conv)]

    assert "你好呀" in sentences


@pytest.mark.asyncio
async def test_tts_failure_skips_audio_but_keeps_text_and_history():
    mock_llm = AsyncMock()
    mock_llm.stream_chat = MagicMock(return_value=_async_iter(["第一句。", "第二句。"])
    )
    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(
        side_effect=[RuntimeError("tts offline"), b"audio"]
    )
    mock_player = AsyncMock()
    mock_player.play_wav_bytes = AsyncMock()

    orch = Orchestrator(mock_llm, mock_tts, mock_player)
    conv = Conversation()

    sentences = [s async for s in orch.chat("hi", conv)]

    assert sentences == ["第一句。", "第二句。"]
    assert mock_player.play_wav_bytes.call_count == 1
    assert conv.get_messages()[-1] == {"role": "assistant", "content": "第一句。第二句。"}


@pytest.mark.asyncio
async def test_empty_llm_response_is_not_saved():
    mock_llm = AsyncMock()
    mock_llm.stream_chat = MagicMock(return_value=_async_iter([]))
    mock_tts = AsyncMock()
    mock_player = AsyncMock()

    orch = Orchestrator(mock_llm, mock_tts, mock_player)
    conv = Conversation()

    with pytest.raises(RuntimeError, match="empty"):
        _ = [s async for s in orch.chat("hi", conv)]

    assert conv.get_messages() == []
