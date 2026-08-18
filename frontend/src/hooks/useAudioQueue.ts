import { useCallback, useEffect, useRef, useState } from "react";
import { selectAutoPlayItem, type QueueItem } from "./audioQueuePolicy";

export type { QueueItem } from "./audioQueuePolicy";

const BAR_COUNT = 24;
const MIN_BAR = 4;

function createAudioContext(): AudioContext | null {
  const Ctor =
    window.AudioContext ??
    (window as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  return Ctor ? new Ctor() : null;
}

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function useAudioQueue() {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [waveform, setWaveform] = useState<number[]>(() =>
    Array.from({ length: BAR_COUNT }, () => MIN_BAR),
  );
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const itemsRef = useRef<QueueItem[]>([]);
  const activeRef = useRef<string | null>(null);

  useEffect(() => {
    itemsRef.current = items;
  }, [items]);
  useEffect(() => {
    activeRef.current = activeKey;
  }, [activeKey]);

  // 一次性初始化：audio 元素 + 音频图（source→analyser→destination）。
  useEffect(() => {
    const audio = new Audio();
    audio.preload = "auto";
    audioRef.current = audio;
    const ctx = createAudioContext();
    ctxRef.current = ctx;
    if (ctx) {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 64;
      analyser.smoothingTimeConstant = 0.8;
      const source = ctx.createMediaElementSource(audio);
      source.connect(analyser);
      analyser.connect(ctx.destination); // 不连 destination 会完全无声
      analyserRef.current = analyser;
    }
    audio.addEventListener("durationchange", () => setDuration(audio.duration || 0));
    audio.addEventListener("timeupdate", () => setCurrentTime(audio.currentTime));
    audio.addEventListener("ended", () => playNextRef.current());
    audio.addEventListener("error", () => {
      setActiveKey(null);
      setIsPlaying(false);
    });
    return () => {
      audio.pause();
      audio.removeAttribute("src");
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      void ctx?.close();
    };
  }, []);

  const playItem = useCallback((item: QueueItem) => {
    const audio = audioRef.current;
    if (!audio || item.url === "") return;
    const ctx = ctxRef.current;
    if (ctx && ctx.state === "suspended") void ctx.resume(); // 用户手势解锁
    setCurrentTime(0);
    setActiveKey(item.key);
    audio.src = item.url;
    audio.play()
      .then(() => setIsPlaying(true))
      .catch(() => setIsPlaying(false)); // 自动播放被拦截：停留为暂停态，点击可重播
  }, []);

  const playNextRef = useRef<() => void>(() => {});
  playNextRef.current = () => {
    const list = itemsRef.current;
    const cur = activeRef.current;
    const index = list.findIndex((item) => item.key === cur);
    const next = list[index + 1];
    if (next) playItem(next);
    else {
      setActiveKey(null);
      setIsPlaying(false);
    }
  };

  const enqueue = useCallback((item: QueueItem) => {
    setItems((prev) =>
      prev.some((existing) => existing.key === item.key) ? prev : [...prev, item],
    );
  }, []);

  // 空闲时从最新入队的语音开始；旧语音仍留在历史消息中供手动重播。
  // 播放中/用户手动暂停时 activeRef 非 null 会跳过，不打断用户控制。
  useEffect(() => {
    const next = selectAutoPlayItem(
      items,
      activeRef.current,
      audioRef.current?.paused ?? false,
    );
    if (next) playItem(next);
  }, [items, playItem]);

  const toggle = useCallback(
    (item: QueueItem) => {
      if (activeRef.current === item.key && audioRef.current && !audioRef.current.paused) {
        audioRef.current.pause();
        setIsPlaying(false);
        return;
      }
      playItem(item);
    },
    [playItem],
  );

  // 波形采样：仅播放中持续 rAF
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    const tick = () => {
      const analyser = analyserRef.current;
      if (analyser) {
        const data = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(data);
        setWaveform(
          Array.from({ length: BAR_COUNT }, (_, i) => {
            const source = Math.floor((i / BAR_COUNT) * data.length);
            return Math.max(MIN_BAR, Math.round((data[source] / 255) * 100));
          }),
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying]);

  return { enqueue, toggle, activeKey, isPlaying, waveform, duration, currentTime, formatTime };
}
