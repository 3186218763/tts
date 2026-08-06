"""Local faster-whisper transcription with lazy model loading."""

from __future__ import annotations

import asyncio
import importlib.util
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


class WhisperTranscriber:
    """Transcribe short browser recordings without blocking the event loop."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        model_factory: Callable[..., Any] | None = None,
    ):
        if not model_size.strip():
            raise ValueError("model_size must not be empty")
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model_factory = model_factory
        self._model = None
        self._model_lock = asyncio.Lock()
        self._cpu_fallback_attempted = False

    @property
    def available(self) -> bool:
        """Whether the importable runtime dependency is present."""
        return self._model_factory is not None or importlib.util.find_spec(
            "faster_whisper"
        ) is not None

    async def _get_model(self):
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    self._create_model, self._device, self._compute_type
                )
        return self._model

    def _create_model(self, device: str, compute_type: str):
        if self._model_factory is not None:
            return self._model_factory(
                model_size=self._model_size,
                device=device,
                compute_type=compute_type,
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "语音输入需要 faster-whisper，请安装项目的 asr 依赖"
            ) from exc
        return WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )

    async def _get_cpu_fallback_model(self):
        async with self._model_lock:
            if not self._cpu_fallback_attempted:
                self._cpu_fallback_attempted = True
                self._model = await asyncio.to_thread(
                    self._create_model, "cpu", "int8"
                )
            return self._model

    @staticmethod
    def _can_fallback_to_cpu(error: RuntimeError) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "libcublas",
                "libcudnn",
                "cuda driver",
                "cuda error",
                "cublas",
            )
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "recording.webm",
        language: str | None = None,
    ) -> dict[str, Any]:
        if not audio:
            raise ValueError("audio must not be empty")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".webm", ".ogg", ".mp4", ".m4a", ".mp3"}:
            suffix = ".webm"

        model = await self._get_model()
        with tempfile.NamedTemporaryFile(suffix=suffix) as recording:
            recording.write(audio)
            recording.flush()

            def run_transcription(active_model) -> dict[str, Any]:
                requested_language = (
                    None if language in {None, "", "auto"} else language
                )
                segments, info = active_model.transcribe(
                    recording.name,
                    language=requested_language,
                    beam_size=self._beam_size,
                    vad_filter=True,
                )
                detected_language = str(getattr(info, "language", "") or "")
                parts = [
                    str(getattr(segment, "text", "") or "").strip()
                    for segment in segments
                ]
                separator = "" if detected_language in {"zh", "ja"} else " "
                text = separator.join(part for part in parts if part)
                probability = getattr(info, "language_probability", None)
                duration = getattr(info, "duration", None)
                return {
                    "text": text,
                    "language": detected_language,
                    "language_probability": (
                        float(probability) if probability is not None else None
                    ),
                    "audio_duration": (
                        float(duration) if duration is not None else None
                    ),
                }

            try:
                return await asyncio.to_thread(run_transcription, model)
            except RuntimeError as exc:
                if self._device != "auto" or not self._can_fallback_to_cpu(exc):
                    raise
                fallback_model = await self._get_cpu_fallback_model()
                return await asyncio.to_thread(run_transcription, fallback_model)
