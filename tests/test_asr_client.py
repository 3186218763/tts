from types import SimpleNamespace

import pytest

from dialogue.asr_client import WhisperTranscriber


class FakeModel:
    def transcribe(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        return (
            [SimpleNamespace(text="  你好 "), SimpleNamespace(text="世界。")],
            SimpleNamespace(language="zh", language_probability=0.98),
        )


@pytest.mark.asyncio
async def test_whisper_transcriber_returns_normalized_text_and_language(tmp_path):
    model = FakeModel()

    def factory(*args, **kwargs):
        return model

    transcriber = WhisperTranscriber(
        model_size="tiny",
        device="cpu",
        model_factory=factory,
    )
    result = await transcriber.transcribe(b"RIFF fake audio", filename="voice.wav")

    assert result["text"] == "你好世界。"
    assert result["language"] == "zh"
    assert result["language_probability"] == pytest.approx(0.98)
    assert result["audio_duration"] is None
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["beam_size"] == 5


@pytest.mark.asyncio
async def test_whisper_transcriber_rejects_empty_audio():
    transcriber = WhisperTranscriber(model_factory=lambda **kwargs: FakeModel())

    with pytest.raises(ValueError, match="audio"):
        await transcriber.transcribe(b"")


@pytest.mark.asyncio
async def test_auto_device_falls_back_to_cpu_when_cuda_runtime_is_missing():
    devices = []

    class MissingCudaModel:
        def transcribe(self, path, **kwargs):
            raise RuntimeError("Library libcublas.so.12 is not found")

    def factory(*, model_size, device, compute_type):
        devices.append((device, compute_type))
        return MissingCudaModel() if device == "auto" else FakeModel()

    transcriber = WhisperTranscriber(
        model_size="tiny",
        device="auto",
        compute_type="auto",
        model_factory=factory,
    )
    result = await transcriber.transcribe(b"RIFF fake audio", filename="voice.wav")

    assert result["text"] == "你好世界。"
    assert devices == [("auto", "auto"), ("cpu", "int8")]
