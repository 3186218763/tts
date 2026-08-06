"""本地 Web 对话入口。

FastAPI 和 Uvicorn 只在启动 Web 时加载，数据管线和单元测试不需要 Web 运行时依赖。
"""

import asyncio
import base64
import json
import re
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from config import AppConfig, load_config
from dialogue.asr_client import WhisperTranscriber
from dialogue.conversation import Conversation
from dialogue.llm_client import LLMClient
from dialogue.persona import get_system_prompt
from dialogue.sentence_streamer import SentenceStreamer
from dialogue.tts_client import TTSClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_HTML = PROJECT_ROOT / "frontend" / "web.html"
MAX_MESSAGE_CHARS = 2000
MAX_SESSION_ID_CHARS = 64
MAX_AUDIO_UPLOAD_BYTES = 15 * 1024 * 1024
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
AUDIO_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
}


def parse_chat_request(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Validate and normalize a chat JSON body."""
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    message = message.strip()
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters")

    session_id = payload.get("session_id", "default")
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string")
    session_id = session_id.strip() or "default"
    if len(session_id) > MAX_SESSION_ID_CHARS or not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            "session_id must contain only letters, numbers, '-' or '_' "
            f"and be at most {MAX_SESSION_ID_CHARS} characters"
        )
    return message, session_id


def sse_event(payload: Mapping[str, Any]) -> str:
    """Encode one JSON event for a browser fetch reader."""
    return f"data: {json.dumps(dict(payload), ensure_ascii=False)}\n\n"


class WebChatService:
    """Stream sentence and audio events while keeping one conversation history."""

    def __init__(self, llm_client, tts_client, *, max_chars: int = 25):
        self._llm = llm_client
        self._tts = tts_client
        self._max_chars = max_chars

    async def stream(
        self, user_text: str, conversation: Conversation
    ) -> AsyncIterator[dict[str, Any]]:
        conversation.add_user_message(user_text)
        messages = [{"role": "system", "content": get_system_prompt()}]
        messages.extend(conversation.get_messages())

        streamer = SentenceStreamer(self._max_chars)
        full_response = ""
        pending_audio = None

        try:
            async for token in self._llm.stream_chat(messages):
                full_response += token
                for sentence in streamer.add_token(token):
                    if pending_audio is not None:
                        yield await self._audio_event(pending_audio)
                    yield {"type": "sentence", "text": sentence}
                    pending_audio = asyncio.create_task(self._tts.synthesize(sentence))

            remaining = streamer.flush()
            if remaining:
                if pending_audio is not None:
                    yield await self._audio_event(pending_audio)
                yield {"type": "sentence", "text": remaining}
                pending_audio = asyncio.create_task(self._tts.synthesize(remaining))
            if pending_audio is not None:
                yield await self._audio_event(pending_audio)
            if not full_response.strip():
                raise RuntimeError("LLM returned an empty response")
        except Exception as exc:
            if pending_audio is not None:
                pending_audio.cancel()
            conversation.rollback_last_user_message()
            yield {"type": "error", "message": str(exc)}
            return

        conversation.add_assistant_message(full_response)
        yield {"type": "done"}

    async def _audio_event(self, task) -> dict[str, str]:
        try:
            audio = await task
        except Exception as exc:
            return {"type": "audio_error", "message": str(exc)}
        return {"type": "audio", "audio": base64.b64encode(audio).decode("ascii")}


def _default_service(config: AppConfig | None = None) -> WebChatService:
    config = config or load_config()
    return WebChatService(
        LLMClient(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
        ),
        TTSClient(
            base_url=config.tts.base_url,
            ref_audio_path=config.tts.ref_audio_path,
            ref_text=config.tts.ref_text,
            ref_language=config.tts.ref_language,
            text_language=config.tts.text_language,
            top_k=config.tts.top_k,
            top_p=config.tts.top_p,
            temperature=config.tts.temperature,
            repetition_penalty=config.tts.repetition_penalty,
            speed_factor=config.tts.speed_factor,
            seed=config.tts.seed,
        ),
        max_chars=config.max_sentence_chars,
    )


def _is_configured(value: str | None, placeholder: str) -> bool:
    return bool(value and value.strip() and value.strip() != placeholder)


def create_app(
    service: WebChatService | None = None,
    *,
    config: AppConfig | None = None,
    transcriber=None,
    max_sessions: int = 128,
):
    """Create the FastAPI app and keep configuration/model loading explicit."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "Web 入口需要额外依赖，请运行：pip install -e '.[web]'"
        ) from exc

    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    runtime_config = config
    if service is None and runtime_config is None:
        runtime_config = load_config()
    chat_service = service or _default_service(runtime_config)
    speech_transcriber = transcriber
    if speech_transcriber is None and runtime_config is not None:
        speech_transcriber = WhisperTranscriber(
            model_size=runtime_config.asr.model,
            device=runtime_config.asr.device,
            compute_type=runtime_config.asr.compute_type,
            beam_size=runtime_config.asr.beam_size,
        )
    max_audio_upload_bytes = MAX_AUDIO_UPLOAD_BYTES
    default_asr_language = "auto"
    if runtime_config is not None:
        max_audio_upload_bytes = max(
            1, int(runtime_config.asr.max_upload_mb * 1024 * 1024)
        )
        default_asr_language = runtime_config.asr.language
    app = FastAPI(title="AI 花音", version="0.1.0")
    conversations: dict[str, Conversation] = {}

    def get_conversation(session_id: str) -> Conversation:
        conversation = conversations.get(session_id)
        if conversation is None:
            if len(conversations) >= max_sessions:
                conversations.pop(next(iter(conversations)))
            conversation = conversations[session_id] = Conversation(
                max_turns=(runtime_config.max_turns if runtime_config else 10)
            )
        return conversation

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return WEB_HTML.read_text(encoding="utf-8")

    @app.get("/healthz")
    async def healthz():
        tts_available = None
        tts_checker = getattr(chat_service, "_tts", None)
        tts_checker = getattr(tts_checker, "check_available", None)
        if callable(tts_checker):
            try:
                await tts_checker()
            except Exception:
                tts_available = False
            else:
                tts_available = True
        return {
            "status": "ok",
            "service": "ai-huayin-web",
            "llm_configured": bool(
                runtime_config
                and _is_configured(runtime_config.llm.api_key, "sk-your-deepseek-api-key")
            ),
            "tts_configured": bool(
                runtime_config
                and _is_configured(runtime_config.tts.ref_text, "在这里填写参考音频对应的文字内容")
            ),
            "tts_available": tts_available,
            "asr_configured": speech_transcriber is not None,
            "asr_available": bool(
                speech_transcriber
                and getattr(speech_transcriber, "available", True)
            ),
        }

    @app.post("/api/chat")
    async def chat(request: Request):
        try:
            payload = await request.json()
            message, session_id = parse_chat_request(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        conversation = get_conversation(session_id)

        async def events():
            async for event in chat_service.stream(message, conversation):
                yield sse_event(event)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/transcribe")
    async def transcribe_audio(request: Request, language: str | None = None):
        if speech_transcriber is None:
            return JSONResponse(
                {"error": "ASR is not configured"},
                status_code=503,
            )
        media_type = request.headers.get("content-type", "").partition(";")[0].lower()
        suffix = AUDIO_SUFFIXES.get(media_type)
        if suffix is None:
            return JSONResponse(
                {"error": "content-type must be a supported audio format"},
                status_code=400,
            )
        try:
            declared_length = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared_length = 0
        if declared_length > max_audio_upload_bytes:
            return JSONResponse(
                {"error": "audio upload exceeds the configured limit"},
                status_code=413,
            )
        audio = await request.body()
        if not audio:
            return JSONResponse({"error": "audio must not be empty"}, status_code=400)
        if len(audio) > max_audio_upload_bytes:
            return JSONResponse(
                {"error": "audio upload exceeds the configured limit"},
                status_code=413,
            )
        language = language or default_asr_language
        if language not in {None, "", "auto", "zh", "ja", "en"}:
            return JSONResponse({"error": "unsupported language"}, status_code=400)
        try:
            result = await speech_transcriber.transcribe(
                audio,
                filename=f"recording{suffix}",
                language=language,
            )
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"transcription failed: {exc}"}, status_code=500)
        if not str(result.get("text") or "").strip():
            return JSONResponse({"error": "no speech recognized"}, status_code=422)
        return result

    @app.post("/api/reset")
    async def reset(request: Request):
        try:
            payload = await request.json()
            session_id = payload.get("session_id", "default")
            _, session_id = parse_chat_request({"message": "placeholder", "session_id": session_id})
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        conversations.pop(session_id, None)
        return {"status": "ok"}

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Start the AI Huayin web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Web 入口需要额外依赖，请运行：pip install -e '.[web]'") from exc
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
