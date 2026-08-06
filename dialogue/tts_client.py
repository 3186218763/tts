"""GPT-SoVITS TTS API 客户端——统一接口，后续可换引擎。"""

import httpx


class TTSClient:
    """调用 GPT-SoVITS API Server 将文本合成为语音。"""

    def __init__(
        self,
        base_url: str,
        ref_audio_path: str,
        ref_text: str,
        ref_language: str,
    ):
        self._base_url = base_url.rstrip("/")
        self._ref_audio_path = ref_audio_path
        self._ref_text = ref_text
        self._ref_language = ref_language

    async def check_available(self) -> None:
        """Raise a user-facing error unless the local API responds."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._base_url}/docs", timeout=5.0)
                response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "TTS 服务未运行，请先启动 GPT-SoVITS API"
            ) from exc

    async def synthesize(self, text: str, text_language: str = "auto") -> bytes:
        """将文本合成为 WAV 格式音频字节。"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/tts",
                json={
                    "text": text,
                    "text_lang": text_language,
                    "ref_audio_path": self._ref_audio_path,
                    "prompt_text": self._ref_text,
                    "prompt_lang": self._ref_language,
                    "text_split_method": "cut0",
                    "media_type": "wav",
                    "streaming_mode": False,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.content
