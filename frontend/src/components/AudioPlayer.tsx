import styles from "./AudioPlayer.module.css";

interface AudioPlayerProps {
  item: { url: string; error?: string };
  active: boolean;
  isPlaying: boolean;
  waveform: number[];
  currentTime: number;
  duration: number;
  formatTime: (seconds: number) => string;
  onToggle: () => void;
}

export function AudioPlayer({
  item, active, isPlaying, waveform, currentTime, duration, formatTime, onToggle,
}: AudioPlayerProps) {
  if (item.error) {
    return <div className={styles.error}>语音合成失败</div>;
  }
  return (
    <button
      type="button"
      className={styles.player}
      onClick={onToggle}
      title={active && isPlaying ? "暂停" : "播放"}
      aria-label={active && isPlaying ? "暂停" : "播放"}
    >
      <span
        className={`${styles.playBtn} ${active && isPlaying ? styles.pause : ""}`}
        aria-hidden="true"
      />
      <span className={styles.bars}>
        {Array.from({ length: waveform.length }, (_, i) => (
          <span
            key={i}
            className={styles.bar}
            style={{
              height: `${active && isPlaying ? waveform[i] : 20 + Math.sin(i * 1.3) * 10}%`,
            }}
          />
        ))}
      </span>
      <span className={styles.time}>
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
    </button>
  );
}
