import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../types";
import { fetchHealth, resetSession, streamChat } from "../api/client";
import type { QueueItem } from "./useAudioQueue";

export type Status =
  | "online"
  | "generating"
  | "recording"
  | "transcribing"
  | "needs-key"
  | "tts-down"
  | "asr-disabled"
  | "mic-error"
  | "insecure"
  | "asr-error"
  | "error";

export interface UseChatOptions {
  enqueueAudio: (item: QueueItem) => void;
}

const SESSION_KEY = "huayin-session-id";

function newId(): string {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function loadSessionId(): string {
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const created = newId();
  localStorage.setItem(SESSION_KEY, created);
  return created;
}

export function useChat({ enqueueAudio }: UseChatOptions) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Status>("online");
  const [asrEnabled, setAsrEnabled] = useState(true);

  const sessionIdRef = useRef(loadSessionId());
  const busyRef = useRef(false);
  const messagesRef = useRef<ChatMessage[]>([]);
  const listRef = useRef<HTMLDivElement | null>(null);
  const nearBottomRef = useRef(true);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // 健康检查 → 状态徽标与录音可用性
  useEffect(() => {
    void fetchHealth()
      .then((health) => {
        if (!health.llm_configured) setStatus("needs-key");
        else if (health.tts_available === false) setStatus("tts-down");
        else if (!health.asr_available) setStatus("asr-disabled");
        else setStatus("online");
        setAsrEnabled(health.asr_available);
      })
      .catch(() => {});
  }, []);

  const onListScroll = useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    nearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  // 新消息自动滚底；用户上翻浏览历史时不强制拉回
  useEffect(() => {
    if (!nearBottomRef.current) return;
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const send = useCallback(
    async (raw: string) => {
      const message = raw.trim();
      if (!message || busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      setStatus("generating");
      const userMessage: ChatMessage = {
        id: newId(),
        role: "user",
        text: message,
        audio: [],
      };
      const assistantId = newId();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        text: "",
        audio: [],
      };
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      try {
        for await (const event of streamChat(message, sessionIdRef.current)) {
          if (event.type === "sentence") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, text: m.text + event.text } : m,
              ),
            );
          } else if (event.type === "audio") {
            const current = messagesRef.current.find((m) => m.id === assistantId);
            const index = current?.audio.length ?? 0;
            const url = `data:audio/wav;base64,${event.audio}`;
            enqueueAudio({ key: `${assistantId}:${index}`, url });
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, audio: [...m.audio, { url }] } : m,
              ),
            );
          } else if (event.type === "audio_error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, audio: [...m.audio, { url: "", error: event.message }] }
                  : m,
              ),
            );
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        }
        setStatus("online");
      } catch (error) {
        const reason = error instanceof Error ? error.message : "未知错误";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  text: m.text ? `${m.text}\n[${reason}]` : `请求失败：${reason}`,
                }
              : m,
          ),
        );
        setStatus("error");
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [enqueueAudio],
  );

  const reset = useCallback(async () => {
    await resetSession(sessionIdRef.current).catch(() => {});
    setMessages([]);
  }, []);

  return { messages, busy, status, asrEnabled, send, reset, listRef, onListScroll };
}
