import { useCallback, useEffect, useState } from "react";
import { useAudioQueue } from "./hooks/useAudioQueue";
import { useChat } from "./hooks/useChat";
import { useRecorder } from "./hooks/useRecorder";
import type { ChatMessage as ChatMessageModel } from "./types";
import { Sidebar } from "./components/Sidebar";
import { StatusBadge } from "./components/StatusBadge";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import type { Status } from "./hooks/useChat";
import styles from "./App.module.css";

export default function App() {
  const audio = useAudioQueue();
  const chat = useChat({ enqueueAudio: audio.enqueue });
  const recorder = useRecorder({ enabled: chat.asrEnabled, onTranscript: chat.send });
  const [drawerOpen, setDrawerOpen] = useState(false);

  // ESC 关闭抽屉（规格 §4.2 抽屉交互规范）
  useEffect(() => {
    if (!drawerOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawerOpen]);

  // 徽标显示优先级：录音错误 > 转写/录音中 > 生成状态
  const displayStatus: Status = recorder.error === "mic" ? "mic-error"
    : recorder.error === "insecure" ? "insecure"
    : recorder.error === "asr" ? "asr-error"
    : recorder.isTranscribing ? "transcribing"
    : recorder.isRecording ? "recording"
    : chat.status;

  const toggleAudio = useCallback((message: ChatMessageModel, index: number) => {
    const url = message.audio[index]?.url;
    if (url) audio.toggle({ key: `${message.id}:${index}`, url });
  }, [audio]);

  return (
    <div className={styles.app}>
      <div className={styles.glow} aria-hidden="true" />
      <Sidebar
        onReset={chat.reset}
        disabled={chat.busy}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
      <main className={styles.main}>
        <header className={styles.topbar}>
          <button
            type="button"
            className={styles.drawerBtn}
            onClick={() => setDrawerOpen(true)}
            aria-label="打开角色信息"
            aria-expanded={drawerOpen}
          >
            <span className={styles.drawerAva}>菜</span>
          </button>
          <div className={styles.brand}>AI 花音</div>
          <StatusBadge status={displayStatus} />
        </header>
        <div className={styles.list} ref={chat.listRef} onScroll={chat.onListScroll}>
          {chat.messages.length === 0 && (
            <div className={styles.empty}>你好，今天想聊点什么？</div>
          )}
          {chat.messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              activeKey={audio.activeKey}
              isPlaying={audio.isPlaying}
              waveform={audio.waveform}
              currentTime={audio.currentTime}
              duration={audio.duration}
              formatTime={audio.formatTime}
              onToggleAudio={toggleAudio}
            />
          ))}
        </div>
        <div className={styles.inputWrap}>
          <ChatInput
            busy={chat.busy}
            asrEnabled={chat.asrEnabled}
            recording={recorder.isRecording}
            onSend={chat.send}
            onToggleRecord={recorder.toggle}
          />
        </div>
      </main>
    </div>
  );
}
