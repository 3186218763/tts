import { useCallback, useEffect, useRef, useState } from "react";
import { transcribeAudio } from "../api/client";

const RECORD_LIMIT_MS = 30_000;
const SUPPORTED_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
];

export type RecorderError = "mic" | "insecure" | "asr" | null;

export interface UseRecorderOptions {
  enabled: boolean;
  onTranscript: (text: string) => void;
}

export function useRecorder({ enabled, onTranscript }: UseRecorderOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<RecorderError>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const timerRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startingRef = useRef(false);

  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
    },
    [],
  );

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  const start = useCallback(async () => {
    if (!enabled || recorderRef.current || startingRef.current) return;
    if (!window.isSecureContext) {
      setError("insecure");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError("mic");
      return;
    }
    startingRef.current = true;
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const mimeType = SUPPORTED_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.addEventListener("dataavailable", (event: BlobEvent) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      });
      recorder.addEventListener("stop", () => {
        if (timerRef.current !== null) window.clearTimeout(timerRef.current);
        // 事件处理器仅在 getUserMedia 成功后才注册，流必然存在
        stream!.getTracks().forEach((track) => track.stop());
        recorderRef.current = null;
        setIsRecording(false);
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        if (blob.size === 0) return;
        setIsTranscribing(true);
        void transcribeAudio(blob)
          .then((text) => {
            setError(null);
            onTranscript(text);
          })
          .catch(() => setError("asr"))
          .finally(() => setIsTranscribing(false));
      });
      recorder.start(250);
      setIsRecording(true);
      setError(null);
      timerRef.current = window.setTimeout(() => {
        if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      }, RECORD_LIMIT_MS);
    } catch {
      stream?.getTracks().forEach((track) => track.stop());
      recorderRef.current = null;
      setIsRecording(false);
      setError("mic");
    } finally {
      startingRef.current = false;
    }
  }, [enabled, onTranscript]);

  const toggle = useCallback(() => {
    if (recorderRef.current?.state === "recording") stop();
    else void start();
  }, [stop, start]);

  return { isRecording, isTranscribing, error, toggle };
}
