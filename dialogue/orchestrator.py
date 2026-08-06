"""对话流水线协调器——LLM 流式生成、句子切分、TTS 合成、音频播放的并行协调。"""

import asyncio
from collections.abc import AsyncIterator

from .conversation import Conversation
from .persona import get_system_prompt
from .sentence_streamer import SentenceStreamer


class Orchestrator:
    """协调 LLM、TTS 和音频播放的流水线。

    用户输入文字后，LLM 流式生成回复，回复被切分为句子，
    每个句子立即送 TTS 合成并播放（后台 worker 顺序播放）。
    chat() 逐句 yield 文字供前端显示，音频在后台并行播放。
    """

    def __init__(self, llm_client, tts_client, audio_player, max_chars: int = 25):
        self._llm = llm_client
        self._tts = tts_client
        self._player = audio_player
        self._max_chars = max_chars

    async def chat(
        self, user_text: str, conversation: Conversation
    ) -> AsyncIterator[str]:
        """处理用户输入，逐句 yield 文字（供显示），后台播放音频。"""
        conversation.add_user_message(user_text)

        messages = [{"role": "system", "content": get_system_prompt()}]
        messages.extend(conversation.get_messages())

        streamer = SentenceStreamer(self._max_chars)
        full_response = ""

        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def tts_worker():
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                try:
                    audio = await self._tts.synthesize(sentence)
                    await self._player.play_wav_bytes(audio)
                except Exception:
                    # A TTS outage must not discard the text response or stop
                    # later sentences from being processed.
                    continue

        worker_task = asyncio.create_task(tts_worker())

        try:
            async for token in self._llm.stream_chat(messages):
                full_response += token
                for sentence in streamer.add_token(token):
                    await tts_queue.put(sentence)
                    yield sentence

            remaining = streamer.flush()
            if remaining:
                await tts_queue.put(remaining)
                yield remaining
            if not full_response.strip():
                raise RuntimeError("LLM returned an empty response")
        except Exception:
            conversation.rollback_last_user_message()
            raise
        finally:
            await tts_queue.put(None)
            await worker_task

        conversation.add_assistant_message(full_response)
