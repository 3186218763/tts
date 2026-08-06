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
        text_language: str = "auto",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 0.85,
        repetition_penalty: float = 1.35,
        speed_factor: float = 1.0,
        seed: int = 42,
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if speed_factor <= 0:
            raise ValueError("speed_factor must be positive")
        self._base_url = base_url.rstrip("/")
        self._ref_audio_path = ref_audio_path
        self._ref_text = ref_text
        self._ref_language = ref_language
        self._text_language = text_language
        self._top_k = top_k
        self._top_p = top_p
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._speed_factor = speed_factor
        self._seed = seed

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

    async def synthesize(self, text: str, text_language: str | None = None) -> bytes:
        """将文本合成为 WAV 格式音频字节。"""
        language = text_language or self._text_language
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/tts",
                json={
                    "text": text,
                    "text_lang": language,
                    "ref_audio_path": self._ref_audio_path,
                    "prompt_text": self._ref_text,
                    "prompt_lang": self._ref_language,
                    "text_split_method": "cut0",
                    "top_k": self._top_k,
                    "top_p": self._top_p,
                    "temperature": self._temperature,
                    "repetition_penalty": self._repetition_penalty,
                    "speed_factor": self._speed_factor,
                    "seed": self._seed,
                    "parallel_infer": True,
                    "media_type": "wav",
                    "streaming_mode": False,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.content
