export type ChatEvent =
  | { type: "sentence"; text: string }
  | { type: "audio"; audio: string }        // base64 WAV
  | { type: "audio_error"; message: string }
  | { type: "error"; message: string }
  | { type: "done" };

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;                             // 随 sentence 事件累积
  audio: { url: string; error?: string }[]; // data: URL 或合成失败标记
}

export interface HealthStatus {
  status: string;
  service: string;
  llm_configured: boolean;
  tts_configured: boolean;
  tts_available: boolean | null;
  asr_configured: boolean;
  asr_available: boolean;
}
