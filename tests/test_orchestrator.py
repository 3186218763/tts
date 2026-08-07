import asyncio
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


@pytest.mark.asyncio
async def test_stage_directions_are_not_displayed_spoken_or_saved():
    mock_llm = AsyncMock()
    mock_llm.stream_chat = MagicMock(
        return_value=_async_iter(["（脸红）", "你好呀。", "*轻轻点头*"])
    )
    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(return_value=b"audio")
    mock_player = AsyncMock()
    mock_player.play_wav_bytes = AsyncMock()

    conv = Conversation()
    sentences = [
        sentence
        async for sentence in Orchestrator(mock_llm, mock_tts, mock_player).chat(
            "hi", conv
        )
    ]

    assert sentences == ["你好呀。"]
    mock_tts.synthesize.assert_awaited_once_with("你好呀。")
    assert conv.get_messages()[-1] == {"role": "assistant", "content": "你好呀。"}


@pytest.mark.asyncio
async def test_synthesizes_next_sentence_while_previous_audio_is_playing():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(return_value=_async_iter(["第一句话。", "第二句话。"] ))
    second_synthesized = asyncio.Event()
    allow_first_playback_to_finish = asyncio.Event()
    synth_calls = 0

    async def synthesize(sentence):
        nonlocal synth_calls
        synth_calls += 1
        if synth_calls == 2:
            second_synthesized.set()
        return sentence.encode()

    async def play(audio):
        if audio.decode() == "第一句话。":
            await asyncio.wait_for(second_synthesized.wait(), timeout=1)
            allow_first_playback_to_finish.set()

    tts = AsyncMock()
    tts.synthesize = AsyncMock(side_effect=synthesize)
    player = AsyncMock()
    player.play_wav_bytes = AsyncMock(side_effect=play)

    sentences = [
        sentence
        async for sentence in Orchestrator(llm, tts, player).chat(
            "继续", Conversation()
        )
    ]

    assert sentences == ["第一句话。", "第二句话。"]
    assert allow_first_playback_to_finish.is_set()


@pytest.mark.asyncio
async def test_chat_injects_rolling_summary_before_recent_turns():
    llm = AsyncMock()
    llm.summarize_chat = AsyncMock(return_value="用户叫小明，喜欢爵士乐。")
    llm.stream_chat = MagicMock(return_value=_async_iter(["当然记得。"]))
    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=b"audio")
    player = AsyncMock()
    player.play_wav_bytes = AsyncMock()
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=2,
        summary_trigger_chars=10_000,
    )
    conversation.add_user_message("我叫小明，喜欢爵士乐")
    conversation.add_assistant_message("记住啦。")
    conversation.add_user_message("今天天气不错")
    conversation.add_assistant_message("很适合散步。")

    _ = [
        sentence
        async for sentence in Orchestrator(llm, tts, player).chat(
            "还记得我的爱好吗", conversation
        )
    ]

    messages = llm.stream_chat.call_args.args[0]
    assert messages[0]["role"] == "system"
    memory_index = next(
        index
        for index, message in enumerate(messages)
        if "用户叫小明，喜欢爵士乐" in message["content"]
    )
    assert memory_index >= 2, "注入段应位于人设卡与记忆之间"
    assert all(message["role"] == "system" for message in messages[1:memory_index])
    assert [message["content"] for message in messages[memory_index + 1 :]] == [
        "今天天气不错",
        "很适合散步。",
        "还记得我的爱好吗",
    ]


@pytest.mark.asyncio
async def test_closing_stream_mid_response_rolls_back_pending_user():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(
        return_value=_async_iter(["第一句话。", "第二句话。"])
    )
    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=b"audio")
    player = AsyncMock()
    conversation = Conversation()

    stream = Orchestrator(llm, tts, player).chat("问题", conversation)
    assert await anext(stream) == "第一句话。"
    await stream.aclose()

    assert conversation.get_messages() == []
