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
