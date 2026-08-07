"""对话流水线协调器——LLM 流式生成、句子切分、TTS 合成、音频播放的并行协调。"""

import asyncio
from collections.abc import AsyncIterator

from .conversation import Conversation
from .memory import prepare_chat_messages
from .sentence_streamer import SentenceStreamer
from .speech_text import normalize_speech_text


class Orchestrator:
    """协调 LLM、TTS 和音频播放的流水线。

    用户输入文字后，LLM 流式生成回复，回复被切分为句子，
    每个句子立即送 TTS 合成并播放（后台 worker 顺序播放）。
    chat() 逐句 yield 文字供前端显示，音频在后台并行播放。
    """

    def __init__(
        self,
        llm_client,
        tts_client,
        audio_player,
        max_chars: int = 50,
        min_chars: int = 4,
    ):
        self._llm = llm_client
        self._tts = tts_client
        self._player = audio_player
        self._max_chars = max_chars
        self._min_chars = min_chars

    async def chat(
        self, user_text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        """处理用户输入，逐句 yield 文字（供显示），后台播放音频。"""
        conversation.add_user_message(user_text)
        committed = False
        try:
            messages = await prepare_chat_messages(self._llm, conversation)
            streamer = SentenceStreamer(self._max_chars, min_chars=self._min_chars)
            full_response = ""

            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
            audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

            async def tts_worker():
                try:
                    while True:
                        sentence = await tts_queue.get()
                        if sentence is None:
                            break
                        try:
                            audio = await self._tts.synthesize(sentence)
                        except Exception:
                            # TTS failures do not discard the text response or
                            # prevent later sentences from being processed.
                            continue
                        await audio_queue.put(audio)
                finally:
                    await audio_queue.put(None)

            async def playback_worker():
                while True:
                    audio = await audio_queue.get()
                    if audio is None:
                        break
                    try:
                        await self._player.play_wav_bytes(audio)
                    except Exception:
                        continue

            tts_task = asyncio.create_task(tts_worker())
            playback_task = asyncio.create_task(playback_worker())
            response_ready = False
            try:
                async for token in self._llm.stream_chat(messages):
                    full_response += token
                    for raw_sentence in streamer.add_token(token):
                        sentence = normalize_speech_text(raw_sentence)
                        if sentence:
                            await tts_queue.put(sentence)
                            yield sentence

                remaining = streamer.flush()
                if remaining:
                    sentence = normalize_speech_text(remaining)
                    if sentence:
                        await tts_queue.put(sentence)
                        yield sentence
                normalized_response = normalize_speech_text(full_response)
                if normalized_response is None:
                    raise RuntimeError("LLM returned an empty response")
                response_ready = True
            finally:
                if response_ready:
                    await tts_queue.put(None)
                    await asyncio.gather(tts_task, playback_task)
                else:
                    tts_task.cancel()
                    playback_task.cancel()
                    await asyncio.gather(
                        tts_task, playback_task, return_exceptions=True
                    )

            conversation.add_assistant_message(normalized_response)
            committed = True
        finally:
            if not committed:
                conversation.rollback_last_user_message()
