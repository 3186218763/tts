import type { ChatEvent, HealthStatus } from "../types";

/** discriminated-union type guard：JSON.parse 返回 unknown，禁 any 下必须窄化。 */
function isChatEvent(value: unknown): value is ChatEvent {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  switch (v.type) {
    case "sentence":
      return typeof v.text === "string";
    case "audio":
      return typeof v.audio === "string";
    case "audio_error":
    case "error":
      return typeof v.message === "string";
    case "done":
      return true;
    default:
      return false;
  }
}

export async function* streamChat(
  message: string,
  sessionId: string,
): AsyncGenerator<ChatEvent> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error || `请求失败（${response.status}）`);
  }
  const stream = response.body;
  if (!stream) throw new Error("浏览器不支持流式响应");
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      if (!chunk.startsWith("data: ")) continue;
      const event = JSON.parse(chunk.slice(6)) as unknown;
      if (!isChatEvent(event)) throw new Error("未知事件类型");
      yield event;
    }
  }
}

export async function transcribeAudio(blob: Blob): Promise<string> {
  const mediaType = (blob.type || "audio/webm").split(";")[0];
  const response = await fetch("/api/transcribe", {
    method: "POST",
    headers: { "Content-Type": mediaType },
    body: blob,
  });
  const result = (await response.json().catch(() => ({}))) as {
    text?: string;
    error?: string;
  };
  if (!response.ok) throw new Error(result.error || "识别失败");
  const text = result.text ?? "";
  if (!text.trim()) throw new Error("没有识别到语音");
  return text;
}

export async function resetSession(sessionId: string): Promise<void> {
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function fetchHealth(): Promise<HealthStatus> {
  const response = await fetch("/healthz");
  if (!response.ok) throw new Error("健康检查失败");
  return (await response.json()) as HealthStatus;
}
