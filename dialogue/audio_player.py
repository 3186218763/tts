"""流式音频播放。"""

import asyncio
import io

sf = None
sd = None


def _load_audio_modules():
    """Load optional desktop audio dependencies only when playback is requested."""
    global sf, sd
    if sf is None:
        try:
            import soundfile as soundfile
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "音频播放依赖不可用：请安装 libsndfile 和 PortAudio，"
                "或使用 Web 界面在浏览器中播放。"
            ) from exc
        sf = soundfile
    if sd is None:
        try:
            import sounddevice as sounddevice
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "音频播放依赖不可用：请安装 PortAudio，"
                "或使用 Web 界面在浏览器中播放。"
            ) from exc
        sd = sounddevice
    return sf, sd


class AudioPlayer:
    """播放 WAV 格式音频字节，播放完毕后返回。"""

    async def play_wav_bytes(self, wav_bytes: bytes) -> None:
        """解析 WAV 字节并播放。通过 to_thread 避免阻塞事件循环。"""
        soundfile, sounddevice = _load_audio_modules()
        audio_data, sample_rate = soundfile.read(
            io.BytesIO(wav_bytes), dtype="float32"
        )
        await asyncio.to_thread(sounddevice.play, audio_data, sample_rate)
        await asyncio.to_thread(sounddevice.wait)
