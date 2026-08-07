import asyncio
import base64

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from dialogue.conversation import Conversation
from frontend.web import WebChatService, create_app, parse_chat_request, sse_event


def _async_iter(tokens):
    async def _gen():
        for token in tokens:
            yield token

    return _gen()


@pytest.mark.asyncio
async def test_web_chat_streams_text_and_audio_events_in_order():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(return_value=_async_iter(["你好呀。", "今天也要", "加油！"]))
    tts = AsyncMock()
    tts.synthesize = AsyncMock(side_effect=[b"wav-one", b"wav-two"])
    service = WebChatService(llm, tts, max_chars=25)

    events = [event async for event in service.stream("hello", Conversation())]

    assert [event["type"] for event in events] == [
        "sentence",
        "audio",
        "sentence",
        "audio",
        "done",
    ]
    assert events[0]["text"] == "你好呀。"
    assert base64.b64decode(events[1]["audio"]) == b"wav-one"
    assert events[2]["text"] == "今天也要加油！"
    assert base64.b64decode(events[3]["audio"]) == b"wav-two"


@pytest.mark.asyncio
async def test_web_chat_updates_conversation_after_stream():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(return_value=_async_iter(["回复。"]))
    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=b"wav")
    conversation = Conversation()

    events = [event async for event in WebChatService(llm, tts).stream("问题", conversation)]

    assert events[-1] == {"type": "done"}
    assert conversation.get_messages() == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回复。"},
    ]


@pytest.mark.asyncio
async def test_web_tts_failure_emits_error_but_completes_text_stream():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(return_value=_async_iter(["回复。", "继续。"]))
    tts = AsyncMock()
    tts.synthesize = AsyncMock(side_effect=[RuntimeError("tts offline"), b"wav"])
    conversation = Conversation()

    events = [
        event
        async for event in WebChatService(llm, tts).stream("问题", conversation)
    ]

    assert [event["type"] for event in events] == [
        "sentence",
        "audio_error",
        "done",
    ]
    assert events[0]["text"] == "回复。继续。"
    assert conversation.get_messages()[-1] == {
        "role": "assistant",
        "content": "回复。继续。",
    }


@pytest.mark.asyncio
async def test_web_empty_llm_response_emits_error_and_rolls_back():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(return_value=_async_iter([]))
    tts = AsyncMock()
    conversation = Conversation()

    events = [
        event
        async for event in WebChatService(llm, tts).stream("问题", conversation)
    ]

    assert events == [{"type": "error", "message": "LLM returned an empty response"}]
    assert conversation.get_messages() == []


@pytest.mark.asyncio
async def test_web_filters_stage_directions_before_text_and_audio():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(
        return_value=_async_iter(["（心里想着：好紧张。）", "见到你真开心！💕"])
    )
    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=b"wav")
    conversation = Conversation()

    events = [
        event
        async for event in WebChatService(llm, tts).stream("你好", conversation)
    ]

    assert [event["type"] for event in events] == ["sentence", "audio", "done"]
    assert events[0]["text"] == "见到你真开心！"
    tts.synthesize.assert_awaited_once_with("见到你真开心！")
    assert conversation.get_messages()[-1] == {
        "role": "assistant",
        "content": "见到你真开心！",
    }


def test_parse_chat_request_rejects_empty_or_oversized_messages():
    with pytest.raises(ValueError, match="message"):
        parse_chat_request({"message": "   "})
    with pytest.raises(ValueError, match="2000"):
        parse_chat_request({"message": "x" * 2001})


def test_parse_chat_request_normalizes_session_id_and_sse_format():
    assert parse_chat_request({"message": " hi ", "session_id": "abc-123"}) == (
        "hi",
        "abc-123",
    )
    payload = sse_event({"type": "done"})
    assert payload == 'data: {"type": "done"}\n\n'


def test_create_app_serves_ui_health_and_streaming_chat():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class FakeService:
        async def stream(self, message, conversation):
            yield {"type": "sentence", "text": f"收到：{message}"}
            yield {"type": "done"}

    from frontend.web import create_app

    class FakeTranscriber:
        async def transcribe(self, audio, *, filename, language=None):
            assert audio == b"fake-webm"
            assert filename.endswith(".webm")
            return {"text": "浏览器语音", "language": "zh"}

    client = TestClient(create_app(FakeService(), transcriber=FakeTranscriber()))
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["llm_configured"] is False
    assert health.json()["asr_configured"] is True
    assert health.json()["tts_available"] is None
    page = client.get("/")
    assert page.status_code == 200
    assert "AI 花音" in page.text
    assert 'id="record"' in page.text
    assert "/api/transcribe" in page.text

    response = client.post(
        "/api/chat",
        json={"message": "你好", "session_id": "test-session"},
    )
    assert response.status_code == 200
    assert '\"type\": \"sentence\"' in response.text
    assert "收到：你好" in response.text
    assert response.headers["content-type"].startswith("text/event-stream")

    reset = client.post("/api/reset", json={"session_id": "test-session"})
    assert reset.status_code == 200

    transcribe = client.post(
        "/api/transcribe",
        content=b"fake-webm",
        headers={"content-type": "audio/webm"},
    )
    assert transcribe.status_code == 200
    assert transcribe.json() == {"text": "浏览器语音", "language": "zh"}

    invalid = client.post(
        "/api/transcribe",
        content=b"not-audio",
        headers={"content-type": "text/plain"},
    )
    assert invalid.status_code == 400

    oversized = client.post(
        "/api/transcribe",
        content=b"small-body",
        headers={
            "content-type": "audio/webm",
            "content-length": str(16 * 1024 * 1024),
        },
    )
    assert oversized.status_code == 413


@pytest.mark.asyncio
async def test_same_session_chat_requests_are_serialized():
    class BlockingService:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.entered: list[str] = []
            self.first_entered = asyncio.Event()
            self.release_first = asyncio.Event()
            self.conversation = None

        async def stream(self, message, conversation):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entered.append(message)
            self.conversation = conversation
            conversation.add_user_message(message)
            try:
                if message == "first":
                    self.first_entered.set()
                    await self.release_first.wait()
                conversation.add_assistant_message(f"reply-{message}")
                yield {"type": "done"}
            finally:
                self.active -= 1

    service = BlockingService()
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        first = asyncio.create_task(
            client.post(
                "/api/chat",
                json={"message": "first", "session_id": "shared"},
            )
        )
        await asyncio.wait_for(service.first_entered.wait(), timeout=1)
        second = asyncio.create_task(
            client.post(
                "/api/chat",
                json={"message": "second", "session_id": "shared"},
            )
        )
        await asyncio.sleep(0.05)

        assert service.entered == ["first"]
        service.release_first.set()
        responses = await asyncio.gather(first, second)

    assert [response.status_code for response in responses] == [200, 200]
    assert service.max_active == 1
    assert service.entered == ["first", "second"]
    assert service.conversation.get_messages() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply-first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply-second"},
    ]


@pytest.mark.asyncio
async def test_reset_waits_for_an_active_chat_in_the_same_session():
    class BlockingService:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.conversation = None

        async def stream(self, message, conversation):
            self.conversation = conversation
            conversation.add_user_message(message)
            self.entered.set()
            await self.release.wait()
            conversation.add_assistant_message("reply")
            yield {"type": "done"}

    service = BlockingService()
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        chat = asyncio.create_task(
            client.post(
                "/api/chat",
                json={"message": "hello", "session_id": "shared"},
            )
        )
        await asyncio.wait_for(service.entered.wait(), timeout=1)
        reset = asyncio.create_task(
            client.post("/api/reset", json={"session_id": "shared"})
        )
        await asyncio.sleep(0.05)

        assert not reset.done()
        service.release.set()
        chat_response, reset_response = await asyncio.gather(chat, reset)

    assert chat_response.status_code == 200
    assert reset_response.status_code == 200
    assert service.conversation.get_messages() == []


@pytest.mark.asyncio
async def test_session_limit_does_not_evict_an_active_conversation():
    class EvictionProbeService:
        def __init__(self):
            self.first_started = asyncio.Event()
            self.followup_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.conversations = {}

        async def stream(self, message, conversation):
            self.conversations[message] = conversation
            if message == "first":
                self.first_started.set()
                await self.release_first.wait()
            elif message == "followup":
                self.followup_started.set()
            yield {"type": "done"}

    service = EvictionProbeService()
    transport = httpx.ASGITransport(app=create_app(service, max_sessions=1))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        first = asyncio.create_task(
            client.post(
                "/api/chat",
                json={"message": "first", "session_id": "protected"},
            )
        )
        await asyncio.wait_for(service.first_started.wait(), timeout=1)

        other = await client.post(
            "/api/chat",
            json={"message": "other", "session_id": "other"},
        )
        assert other.status_code == 200

        followup = asyncio.create_task(
            client.post(
                "/api/chat",
                json={"message": "followup", "session_id": "protected"},
            )
        )
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    service.followup_started.wait(), timeout=0.05
                )
        finally:
            service.release_first.set()
            first_response, followup_response = await asyncio.gather(
                first, followup
            )

    assert first_response.status_code == 200
    assert followup_response.status_code == 200
    assert service.conversations["first"] is service.conversations["followup"]


@pytest.mark.asyncio
async def test_closing_web_stream_mid_response_rolls_back_pending_user():
    llm = AsyncMock()
    llm.stream_chat = MagicMock(
        return_value=_async_iter(["第一句话。", "第二句话。"])
    )
    tts = AsyncMock()
    conversation = Conversation()

    stream = WebChatService(llm, tts).stream("问题", conversation)
    assert await anext(stream) == {"type": "sentence", "text": "第一句话。"}
    await stream.aclose()

    assert conversation.get_messages() == []
