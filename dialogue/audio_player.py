"""流式音频播放。"""

import asyncio
import io

import sounddevice as sd
import soundfile as sf


class AudioPlayer:
    """播放 WAV 格式音频字节，播放完毕后返回。"""

    async def play_wav_bytes(self, wav_bytes: bytes) -> None:
        """解析 WAV 字节并播放。通过 to_thread 避免阻塞事件循环。"""
        audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        await asyncio.to_thread(sd.play, audio_data, sample_rate)
        await asyncio.to_thread(sd.wait)
