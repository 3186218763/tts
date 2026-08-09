import { useRef, useState } from "react";
import styles from "./ChatInput.module.css";

const MAX_CHARS = 2000;

interface ChatInputProps {
  busy: boolean;
  asrEnabled: boolean;
  recording: boolean;
  onSend: (text: string) => void;
  onToggleRecord: () => void;
}

export function ChatInput({ busy, asrEnabled, recording, onSend, onToggleRecord }: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const submit = () => {
    const value = text.trim();
    if (!value || busy) return;
    setText("");
    onSend(value);
    const el = textareaRef.current;
    if (el) el.style.height = "auto";
  };

  const resize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  };

  return (
    <div className={styles.bar}>
      <button
        type="button"
        className={`${styles.mic} ${recording ? styles.recording : ""}`}
        onClick={onToggleRecord}
        disabled={!asrEnabled || busy}
        title={!asrEnabled ? "语音模型未安装" : recording ? "停止录音" : "开始录音"}
        aria-label={!asrEnabled ? "语音模型未安装" : recording ? "停止录音" : "开始录音"}
      >
        <span className={styles.dot} aria-hidden="true" />
      </button>
      <div className={styles.fieldWrap}>
        <textarea
          ref={textareaRef}
          className={styles.field}
          value={text}
          maxLength={MAX_CHARS}
          placeholder="输入消息...（Enter 发送，Shift+Enter 换行）"
          aria-label="消息"
          onChange={(event) => setText(event.target.value)}
          onInput={resize}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
        />
        {text.length > MAX_CHARS - 100 && (
          <span className={styles.count}>{text.length}/{MAX_CHARS}</span>
        )}
      </div>
      <button
        type="button"
        className={styles.send}
        onClick={submit}
        disabled={busy || !text.trim()}
        title="发送"
        aria-label="发送"
      >
        ↑
      </button>
    </div>
  );
}
