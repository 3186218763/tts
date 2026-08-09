import type { Status } from "../hooks/useChat";
import styles from "./StatusBadge.module.css";

const STATUS_TEXT: Record<Status, string> = {
  online: "在线",
  generating: "生成中…",
  recording: "录音中…",
  transcribing: "识别中…",
  "needs-key": "请配置 DeepSeek API key",
  "tts-down": "TTS 服务未启动",
  "asr-disabled": "在线（语音不可用）",
  "mic-error": "无法访问麦克风",
  insecure: "录音仅支持 localhost 或 HTTPS",
  "asr-error": "识别失败",
  error: "连接异常",
};

const STATUS_TONE: Record<Status, string> = {
  online: styles.ok,
  "asr-disabled": styles.ok,
  "needs-key": styles.warn,
  "tts-down": styles.warn,
  insecure: styles.warn,
  generating: styles.active,
  recording: styles.active,
  transcribing: styles.active,
  "mic-error": styles.err,
  "asr-error": styles.err,
  error: styles.err,
};

export function StatusBadge({ status }: { status: Status }) {
  return (
    <span className={`${styles.badge} ${STATUS_TONE[status]}`}>
      {STATUS_TEXT[status]}
    </span>
  );
}
