import json

import httpx
import pytest
import respx

from dialogue.tts_client import TTSClient


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_returns_audio():
    respx.post("http://127.0.0.1:9880/tts").mock(
        return_value=httpx.Response(200, content=b"fake_wav_bytes")
    )
    client = TTSClient(
        base_url="http://127.0.0.1:9880",
        ref_audio_path="/ref.wav",
        ref_text="参考文本",
        ref_language="zh",
    )
    result = await client.synthesize("你好世界", "zh")
    assert result == b"fake_wav_bytes"


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_sends_correct_body():
    route = respx.post("http://127.0.0.1:9880/tts").mock(
        return_value=httpx.Response(200, content=b"audio")
    )
    client = TTSClient(
        base_url="http://127.0.0.1:9880",
        ref_audio_path="/ref.wav",
        ref_text="参考文本",
        ref_language="zh",
    )
    await client.synthesize("你好世界", "zh")

    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["text"] == "你好世界"
    assert body["text_lang"] == "zh"
    assert body["ref_audio_path"] == "/ref.wav"
    assert body["prompt_text"] == "参考文本"
    assert body["prompt_lang"] == "zh"
    assert body["text_split_method"] == "cut0"
    assert body["top_k"] == 15
    assert body["top_p"] == 1.0
    assert body["temperature"] == 0.85
    assert body["repetition_penalty"] == 1.35
    assert body["speed_factor"] == 1.0
    assert body["seed"] == 42
    assert body["parallel_infer"] is True
    assert body["streaming_mode"] is False


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_default_language_is_auto():
    route = respx.post("http://localhost:9880/tts").mock(
        return_value=httpx.Response(200, content=b"audio")
    )
    client = TTSClient("http://localhost:9880", "/r.wav", "ref", "zh")
    await client.synthesize("test")
    body = json.loads(route.calls[0].request.content)
    assert body["text_lang"] == "auto"


@pytest.mark.asyncio
@respx.mock
async def test_check_available_accepts_running_api():
    route = respx.get("http://127.0.0.1:9880/docs").mock(
        return_value=httpx.Response(200, text="ok")
    )
    client = TTSClient("http://127.0.0.1:9880", "/r.wav", "ref", "zh")

    await client.check_available()

    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_check_available_raises_clear_error_when_api_is_down():
    respx.get("http://127.0.0.1:9880/docs").mock(
        return_value=httpx.Response(503)
    )
    client = TTSClient("http://127.0.0.1:9880", "/r.wav", "ref", "zh")

    with pytest.raises(RuntimeError, match="TTS 服务未运行"):
        await client.check_available()
