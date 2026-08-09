import type { ChatMessage as ChatMessageModel } from "../types";
import { AudioPlayer } from "./AudioPlayer";
import styles from "./ChatMessage.module.css";

interface ChatMessageProps {
  message: ChatMessageModel;
  activeKey: string | null;
  isPlaying: boolean;
  waveform: number[];
  currentTime: number;
  duration: number;
  formatTime: (seconds: number) => string;
  onToggleAudio: (message: ChatMessageModel, index: number) => void;
}

export function ChatMessage({
  message, activeKey, isPlaying, waveform, currentTime, duration, formatTime, onToggleAudio,
}: ChatMessageProps) {
  return (
    <div className={`${styles.row} ${message.role === "user" ? styles.user : styles.ai}`}>
      <div className={styles.bubble}>
        {message.text || "…"}
        {message.audio.map((audio, index) => (
          <AudioPlayer
            key={`${message.id}:${index}`}
            item={audio}
            active={activeKey === `${message.id}:${index}`}
            isPlaying={isPlaying}
            waveform={waveform}
            currentTime={currentTime}
            duration={duration}
            formatTime={formatTime}
            onToggle={() => onToggleAudio(message, index)}
          />
        ))}
      </div>
    </div>
  );
}
