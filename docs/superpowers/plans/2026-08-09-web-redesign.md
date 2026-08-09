# Web 前端重设计实现计划（React + TypeScript 暖夜深色版）

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 把单文件 `frontend/web.html`（原生 JS）替换为 Vite + React + TypeScript 工程，暖夜深色视觉（角色侧边栏 + 聊天主区），后端 API 零改动。

**架构：** 前端为独立 npm 工程（`frontend/`），构建产物进 `frontend/dist/`（不提交 git）；`frontend/web.py` 服务 `dist/index.html` 并条件挂载 `/assets` 静态目录，`create_app` 增加 `index_html` 注入参数供测试确定性验证；`tests/test_web.py` 首页断言改为注入式用例。SSE 流式对话、录音转写、会话管理逻辑从现状原样迁移。

**技术栈：** Vite + React 19 + TypeScript 5（`strict`，禁 `any`），CSS Modules + 设计令牌（tokens.css），pytest（后端）。

**测试策略说明：** 前端不引入 vitest（规格 §7 验证手段 = `tsc --noEmit` + `npm run build` + 后端 pytest + 手动验收清单）；后端适配采用 pytest TDD（红→绿）。

**规格：** `docs/superpowers/specs/2026-08-09-web-redesign-design.md`

---

## 文件结构

**前端（全部新建，除 .gitignore 外）：**

| 文件 | 职责 |
|---|---|
| `frontend/package.json` | npm 清单与脚本（dev/build/preview） |
| `frontend/tsconfig.json` | strict TS 配置 |
| `frontend/vite.config.ts` | React 插件 + dev proxy（/api、/healthz → 8000） |
| `frontend/index.html` | 入口模板（`<title>AI 花音</title>` + `<div id="root">`） |
| `frontend/src/main.tsx` | 挂载 React |
| `frontend/src/App.tsx` | 布局组合：Sidebar + 顶栏 + 消息列表 + ChatInput |
| `frontend/src/types.ts` | `ChatEvent` / `ChatMessage` / `HealthStatus` |
| `frontend/src/api/client.ts` | SSE 流式解析（type guard）+ transcribe/reset/healthz |
| `frontend/src/hooks/useAudioQueue.ts` | 单 audio 元素顺序播放 + AnalyserNode 波形 + 时长 |
| `frontend/src/hooks/useRecorder.ts` | MediaRecorder 录音 → 转写 → 回调 |
| `frontend/src/hooks/useChat.ts` | 消息列表 + SSE 消费 + 健康检查 + 自动滚底 |
| `frontend/src/components/Sidebar.tsx` | 角色卡 + 移动端抽屉 |
| `frontend/src/components/StatusBadge.tsx` | 顶部状态徽标（11 态） |
| `frontend/src/components/ChatMessage.tsx` | 气泡 + 内嵌 AudioPlayer 列表 |
| `frontend/src/components/AudioPlayer.tsx` | 播放按钮 + 波形条 + 时长 |
| `frontend/src/components/ChatInput.tsx` | 录音/输入框/发送 |
| `frontend/src/styles/tokens.css` | 设计令牌 + reset + 全局动效 keyframes |
| `frontend/src/*.module.css` ×5 | 组件样式（App/Sidebar/StatusBadge/AudioPlayer/ChatMessage/ChatInput） |

**后端修改：**

| 文件 | 职责 |
|---|---|
| `frontend/web.py` | `WEB_HTML` 指向 dist、`/assets` 条件挂载、`create_app(index_html=...)`、503 提示页 |
| `tests/test_web.py` | 首页断言改注入式两个用例 + src 源码 API 路径检查 |
| `frontend/web.html` | **删除**（被 React 应用替换） |
| `pyproject.toml` | 删除 `[tool.setuptools.package-data] frontend = ["web.html"]` |
| `.gitignore` | 增加 `frontend/dist/`、`frontend/node_modules/` |
| `README.md` | 功能特性/快速开始/项目结构/测试数更新 |

---

### 任务 1：前端工程脚手架

**文件：**
- 创建：`frontend/package.json`
- 创建：`frontend/tsconfig.json`
- 创建：`frontend/vite.config.ts`
- 创建：`frontend/index.html`
- 创建：`frontend/src/main.tsx`
- 创建：`frontend/src/App.tsx`（占位，任务 10 替换）
- 创建：`frontend/src/styles/tokens.css`
- 修改：`.gitignore`（根目录）

- [ ] **步骤 1：写工程配置文件**

`frontend/package.json`：

```json
{
  "name": "ai-huayin-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.1.0",
    "react-dom": "^19.1.0"
  },
  "devDependencies": {
    "@types/react": "^19.1.0",
    "@types/react-dom": "^19.1.0",
    "@vitejs/plugin-react": "^4.5.0",
    "typescript": "^5.8.0",
    "vite": "^7.0.0"
  }
}
```

`frontend/tsconfig.json`：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "noEmit": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
```

`frontend/vite.config.ts`：

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/healthz": "http://127.0.0.1:8000",
    },
  },
});
```

`frontend/index.html`：

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI 花音</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

`.gitignore`（根目录，追加在 `# 运行时日志和生成输出` 段之前）：

```
# 前端构建产物
frontend/dist/
frontend/node_modules/
```

- [ ] **步骤 2：写最小入口与设计令牌**

`frontend/src/main.tsx`：

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/tokens.css";

const root = document.getElementById("root");
if (!root) throw new Error("缺少 #root 挂载节点");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

`frontend/src/App.tsx`（占位，任务 10 替换为完整版）：

```tsx
export default function App() {
  return <div>AI 花音 Web 前端</div>;
}
```

`frontend/src/styles/tokens.css`（设计令牌，完整版）：

```css
:root {
  --bg: #0f1315;
  --panel: rgba(255, 255, 255, 0.03);
  --line: rgba(255, 255, 255, 0.09);
  --text: #f1ece7;
  --muted: #9eaeab;
  --accent: #f2b6a0;
  --accent-strong: #ff917d;
  --accent-gradient: linear-gradient(135deg, #f2b6a0, #ff917d);
  --ai-bubble: linear-gradient(160deg, #2b2430, #26212c);
  --user-bubble: linear-gradient(160deg, #1e3a3d, #1b3437);
  --user-bubble-line: rgba(120, 190, 185, 0.25);
  --ok: #7fcf9a;
  --danger: #ec665d;
  --shadow-panel: 0 8px 26px rgba(0, 0, 0, 0.35);
  --shadow-bubble: 0 4px 14px rgba(0, 0, 0, 0.25);
  --ease: cubic-bezier(0.22, 0.61, 0.36, 1);
  --dur-fast: 160ms;
  --dur-med: 220ms;
}

* { box-sizing: border-box; }
html, body, #root { margin: 0; height: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  overflow: hidden;
}
h1, h2, h3, p { margin: 0; }
button {
  border: 0; color: inherit; font: inherit; cursor: pointer; background: none; padding: 0;
}
button:disabled { cursor: not-allowed; opacity: 0.45; }
textarea { font: inherit; }
```

- [ ] **步骤 3：安装依赖并验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm install && npm run build`
预期：tsc 零错误；`dist/index.html` 与 `dist/assets/*.js` 生成。

- [ ] **步骤 4：Commit**

```bash
cd /home/miku/dv_project/tts && git add .gitignore frontend/ && git commit -m "build: 前端工程脚手架——Vite + React + TypeScript

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 2：类型与 API 客户端

**文件：**
- 创建：`frontend/src/types.ts`
- 创建：`frontend/src/api/client.ts`

- [ ] **步骤 1：写 types.ts**

```ts
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
```

- [ ] **步骤 2：写 api/client.ts（SSE 解析 + type guard）**

```ts
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
```

- [ ] **步骤 3：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 4：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/types.ts frontend/src/api/ && git commit -m "feat: 类型化 API 客户端——SSE 流式解析与 type guard

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 3：useAudioQueue（顺序播放 + 波形可视化）

**文件：**
- 创建：`frontend/src/hooks/useAudioQueue.ts`

要点：**单个隐藏 audio 元素复用**（`createMediaElementSource` 对同一元素只允许一次，重复调用抛 `InvalidStateError`；单元素天然保证「全局最多一个在播」）；音频图连接链 `source → analyser → destination` 缺一不可（不连 destination 完全无声）；首次用户手势 `ctx.resume()` 解锁；`play()` 必须 `.catch`（自动播放策略拦截时停留为暂停态，点击重播）。

- [ ] **步骤 1：写 useAudioQueue.ts**

```ts
import { useCallback, useEffect, useRef, useState } from "react";

const BAR_COUNT = 24;
const MIN_BAR = 4;

export interface QueueItem {
  key: string; // `${messageId}:${index}`
  url: string;
}

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
```

- [ ] **步骤 2：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 3：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/hooks/useAudioQueue.ts && git commit -m "feat: 音频顺序播放队列与波形可视化

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 4：useRecorder（录音 → 转写）

**文件：**
- 创建：`frontend/src/hooks/useRecorder.ts`

- [ ] **步骤 1：写 useRecorder.ts**

```ts
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

  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    },
    [],
  );

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  const start = useCallback(async () => {
    if (!enabled || recorderRef.current) return;
    if (!window.isSecureContext) {
      setError("insecure");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError("mic");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
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
        stream.getTracks().forEach((track) => track.stop());
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
      setError("mic");
    }
  }, [enabled, onTranscript]);

  const toggle = useCallback(() => {
    if (recorderRef.current?.state === "recording") stop();
    else void start();
  }, [stop, start]);

  return { isRecording, isTranscribing, error, toggle };
}
```

- [ ] **步骤 2：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 3：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/hooks/useRecorder.ts && git commit -m "feat: 录音转写 hook——MediaRecorder + secure context 校验

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 5：useChat（消息列表 + SSE 消费 + 健康检查）

**文件：**
- 创建：`frontend/src/hooks/useChat.ts`

- [ ] **步骤 1：写 useChat.ts**

```ts
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
```

- [ ] **步骤 2：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 3：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/hooks/useChat.ts && git commit -m "feat: 对话 hook——SSE 消费、健康检查与自动滚底

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 6：StatusBadge 组件

**文件：**
- 创建：`frontend/src/components/StatusBadge.tsx`
- 创建：`frontend/src/components/StatusBadge.module.css`

- [ ] **步骤 1：写组件**

```tsx
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
```

- [ ] **步骤 2：写样式**

```css
.badge {
  display: inline-flex; align-items: center; gap: 5px; font-size: 11px;
  padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.1); background: rgba(255, 255, 255, 0.04);
}
.badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
.ok { color: #a9c7bb; border-color: rgba(127, 207, 154, 0.25); background: rgba(127, 207, 154, 0.08); }
.ok::before { background: var(--ok); }
.active { color: #f2cfae; border-color: rgba(242, 182, 160, 0.35); background: rgba(242, 182, 160, 0.08); }
.active::before { background: var(--accent); animation: pulse 1.2s ease-in-out infinite; }
.warn { color: #e8c9a2; border-color: rgba(232, 201, 162, 0.3); background: rgba(232, 201, 162, 0.07); }
.warn::before { background: #e8b47a; }
.err { color: #f0a39c; border-color: rgba(236, 102, 93, 0.35); background: rgba(236, 102, 93, 0.08); }
.err::before { background: var(--danger); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
```

- [ ] **步骤 3：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 4：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/components/StatusBadge.tsx frontend/src/components/StatusBadge.module.css && git commit -m "feat: 状态徽标组件（11 态）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 7：AudioPlayer 组件

**文件：**
- 创建：`frontend/src/components/AudioPlayer.tsx`
- 创建：`frontend/src/components/AudioPlayer.module.css`

- [ ] **步骤 1：写组件**

```tsx
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
```

- [ ] **步骤 2：写样式**

```css
.player {
  margin-top: 8px; height: 26px; width: 168px; border-radius: 8px;
  display: flex; align-items: center; gap: 6px; padding: 0 10px;
  background: rgba(242, 182, 160, 0.09); border: 1px solid rgba(242, 182, 160, 0.28);
  transition: border-color var(--dur-fast) var(--ease);
}
.player:hover { border-color: rgba(242, 182, 160, 0.5); }
.playBtn {
  width: 14px; height: 14px; border-radius: 50%; flex: 0 0 14px; display: grid; place-items: center;
  background: var(--accent-gradient); color: #241310; font-size: 7px; font-weight: 700;
  box-shadow: 0 0 6px rgba(255, 145, 125, 0.6);
}
.playBtn::before { content: "▶"; }
.playBtn.pause::before { content: "❚❚"; }
.bars { flex: 1; display: flex; align-items: center; gap: 2.5px; height: 14px; }
.bar { flex: 1; border-radius: 2px; background: linear-gradient(180deg, #ffd9c7, #f2a98c); opacity: 0.85; }
.time { font-size: 10px; color: #e0b9a6; letter-spacing: 0.03em; }
.error { margin-top: 8px; font-size: 11px; color: #d9a09a; }
```

- [ ] **步骤 3：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 4：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/components/AudioPlayer.tsx frontend/src/components/AudioPlayer.module.css && git commit -m "feat: 音频播放器组件（波形条 + 时长）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 8：ChatMessage 与 ChatInput 组件

**文件：**
- 创建：`frontend/src/components/ChatMessage.tsx`
- 创建：`frontend/src/components/ChatMessage.module.css`
- 创建：`frontend/src/components/ChatInput.tsx`
- 创建：`frontend/src/components/ChatInput.module.css`

- [ ] **步骤 1：写 ChatMessage.tsx**

```tsx
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
```

- [ ] **步骤 2：写 ChatMessage.module.css**

```css
.row { display: flex; }
.ai { justify-content: flex-start; }
.user { justify-content: flex-end; }
.bubble {
  max-width: 76%; padding: 10px 14px; border-radius: 14px;
  font-size: 13.5px; white-space: pre-wrap; overflow-wrap: anywhere;
  animation: riseIn var(--dur-med) var(--ease) both;
}
.ai .bubble {
  background: var(--ai-bubble); border: 1px solid rgba(255, 255, 255, 0.09);
  border-bottom-left-radius: 5px; box-shadow: var(--shadow-bubble);
}
.user .bubble {
  background: var(--user-bubble); border: 1px solid var(--user-bubble-line);
  border-bottom-right-radius: 5px; box-shadow: var(--shadow-bubble);
}
@keyframes riseIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
```

- [ ] **步骤 3：写 ChatInput.tsx**

```tsx
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
            if (event.key === "Enter" && !event.shiftKey) {
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
```

- [ ] **步骤 4：写 ChatInput.module.css**

```css
.bar {
  display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 16px;
  background: var(--panel); border: 1px solid rgba(255, 255, 255, 0.09);
  box-shadow: var(--shadow-panel), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.mic {
  width: 38px; height: 38px; border-radius: 50%; flex: 0 0 38px;
  border: 1px solid rgba(255, 255, 255, 0.12); background: rgba(255, 255, 255, 0.03);
  display: grid; place-items: center;
  transition: border-color var(--dur-fast) var(--ease);
}
.mic:hover { border-color: var(--accent); }
.dot {
  width: 11px; height: 11px; border-radius: 50%;
  background: var(--danger); box-shadow: 0 0 8px rgba(236, 102, 93, 0.55);
  transition: all var(--dur-fast) var(--ease);
}
.mic.recording { border-color: var(--danger); }
.mic.recording .dot { border-radius: 2px; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.fieldWrap { flex: 1; min-width: 0; position: relative; }
.field {
  display: block; width: 100%; min-height: 42px; max-height: 140px; resize: vertical;
  border: 0; outline: 0; background: transparent; color: var(--text);
}
.field::placeholder { color: var(--muted); }
.count { position: absolute; right: 4px; bottom: 2px; font-size: 10px; color: #e8b47a; pointer-events: none; }
.send {
  width: 40px; height: 40px; border-radius: 12px; flex: 0 0 40px; display: grid; place-items: center;
  background: var(--accent-gradient); color: #241310; font-size: 20px; font-weight: 700;
  box-shadow: 0 4px 16px rgba(255, 145, 125, 0.35);
  transition: transform var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
.send:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(255, 145, 125, 0.45); }
```

- [ ] **步骤 5：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 6：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/components/ChatMessage.tsx frontend/src/components/ChatMessage.module.css frontend/src/components/ChatInput.tsx frontend/src/components/ChatInput.module.css && git commit -m "feat: 消息气泡与输入栏组件

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 9：Sidebar 组件（角色卡 + 移动端抽屉）

**文件：**
- 创建：`frontend/src/components/Sidebar.tsx`
- 创建：`frontend/src/components/Sidebar.module.css`

- [ ] **步骤 1：写 Sidebar.tsx**

```tsx
import styles from "./Sidebar.module.css";

interface SidebarProps {
  onReset: () => void;
  disabled: boolean;
  open: boolean;
  onClose: () => void;
}

export function Sidebar({ onReset, disabled, open, onClose }: SidebarProps) {
  return (
    <>
      <aside className={`${styles.side} ${open ? styles.open : ""}`} role="complementary">
        <div className={styles.avatar} aria-hidden="true">菜</div>
        <h2 className={styles.name}>真白花音</h2>
        <div className={styles.role}>AI 复刻 · 纪念向</div>
        <span className={styles.status}><span className={styles.dot} />在线 · 深夜电台</span>
        <p className={styles.desc}>
          元气、温柔、笨拙但倔强。2020 年起在 B 站开播，2026-05-01 毕业。
          这里是粉丝为她搭建的 AI 纪念亭——她已不在，但声音还在。
        </p>
        <p className={styles.fans}>「每天都和白菜在一起」</p>
        <div className={styles.spacer} />
        <button type="button" className={styles.reset} onClick={onReset} disabled={disabled}>
          ↻ 清空对话
        </button>
        <p className={styles.note}>AI 复刻纪念项目 · 非本人</p>
      </aside>
      <div
        className={`${styles.overlay} ${open ? styles.overlayVisible : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />
    </>
  );
}
```

- [ ] **步骤 2：写 Sidebar.module.css**

```css
.side {
  width: 262px; flex: 0 0 262px; position: relative; z-index: 2;
  background: rgba(255, 255, 255, 0.022); border-right: 1px solid rgba(255, 255, 255, 0.07);
  display: flex; flex-direction: column; padding: 26px 20px 18px;
}
.avatar {
  width: 72px; height: 72px; border-radius: 50%; margin: 0 auto;
  display: grid; place-items: center; font-size: 30px; font-weight: 700; color: #ffd9c7;
  background: radial-gradient(circle at 35% 30%, #2a2f33, #151a1c); position: relative;
}
.avatar::before {
  content: ""; position: absolute; inset: -5px; border-radius: 50%;
  background: conic-gradient(from 210deg, #f2b6a0, #ff917d, #8a5a4e, #f2b6a0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 2px), #000 calc(100% - 1.5px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 2px), #000 calc(100% - 1.5px));
  filter: drop-shadow(0 0 8px rgba(255, 145, 125, 0.5));
}
.avatar::after {
  content: ""; position: absolute; bottom: -14px; left: 50%; transform: translateX(-50%);
  width: 44px; height: 10px; border-radius: 50%; background: rgba(255, 145, 125, 0.28); filter: blur(6px);
}
.name { margin-top: 22px; text-align: center; font-size: 17px; font-weight: 650; letter-spacing: 0.05em; }
.role { text-align: center; font-size: 11px; color: var(--muted); margin-top: 3px; }
.status {
  margin: 14px auto 0; display: inline-flex; align-items: center; gap: 6px;
  font-size: 11.5px; color: #c8d6d2;
  background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.09);
  padding: 4px 12px; border-radius: 999px;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 8px rgba(127, 207, 154, 0.8); }
.desc {
  margin-top: 22px; font-size: 12px; line-height: 1.75; color: #a9b6b2;
  border-left: 2px solid rgba(242, 182, 160, 0.45); padding-left: 10px;
}
.fans { margin-top: 14px; font-size: 11.5px; color: #e0b9a6; text-align: center; font-style: italic; opacity: 0.85; }
.spacer { flex: 1; }
.reset {
  width: 100%; padding: 9px 0; border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.1); background: rgba(255, 255, 255, 0.03);
  color: var(--muted); font-size: 12.5px;
  transition: all var(--dur-fast) var(--ease);
}
.reset:hover { color: #ffd9c7; border-color: rgba(242, 182, 160, 0.5); background: rgba(242, 182, 160, 0.07); }
.note { margin-top: 10px; text-align: center; font-size: 10px; color: rgba(158, 174, 171, 0.6); }
.overlay { display: none; }

@media (max-width: 768px) {
  .side {
    position: fixed; top: 0; bottom: 0; left: 0;
    transform: translateX(-105%); transition: transform var(--dur-med) var(--ease);
    background: #12161a;
  }
  .side.open { transform: translateX(0); box-shadow: 0 0 60px rgba(0, 0, 0, 0.6); }
  .overlay {
    display: block; position: fixed; inset: 0; z-index: 1;
    background: rgba(0, 0, 0, 0.5); opacity: 0; pointer-events: none;
    transition: opacity var(--dur-med) var(--ease);
  }
  .overlayVisible { opacity: 1; pointer-events: auto; }
}
```

- [ ] **步骤 3：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误。

- [ ] **步骤 4：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/components/Sidebar.tsx frontend/src/components/Sidebar.module.css && git commit -m "feat: 角色侧边栏组件（角色卡 + 移动端抽屉）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 10：App 布局接线

**文件：**
- 修改：`frontend/src/App.tsx`（替换任务 1 的占位版）
- 创建：`frontend/src/App.module.css`

- [ ] **步骤 1：写完整 App.tsx**

```tsx
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
```

- [ ] **步骤 2：写 App.module.css**

```css
.app { display: flex; height: 100vh; width: 100vw; position: relative; overflow: hidden; }
.glow {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(55% 45% at 78% -5%, rgba(255, 160, 110, 0.16), transparent 60%),
    radial-gradient(40% 35% at -5% 100%, rgba(255, 145, 125, 0.08), transparent 60%);
  animation: breathe 8s ease-in-out infinite;
}
@keyframes breathe { 0%, 100% { opacity: 0.75; } 50% { opacity: 1; } }
.main { flex: 1; min-width: 0; display: flex; flex-direction: column; position: relative; z-index: 1; }
.topbar {
  height: 56px; flex: 0 0 56px; display: flex; align-items: center; gap: 14px;
  padding: 0 24px; border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.brand { flex: 1; font-size: 14px; font-weight: 600; letter-spacing: 0.08em; color: #e8d9d0; }
.drawerBtn {
  display: none; width: 38px; height: 38px; border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.12); align-items: center; justify-content: center;
}
.drawerAva {
  width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
  font-size: 12px; font-weight: 700; color: #ffd9c7;
  background: radial-gradient(circle at 35% 30%, #2a2f33, #151a1c);
}
.list { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
.list > :first-child { margin-top: auto; }
.empty { margin: auto; color: var(--muted); text-align: center; font-size: 14px; }
.inputWrap { padding: 0 24px 22px; }

@media (max-width: 768px) {
  .drawerBtn { display: inline-flex; }
  .list { padding: 16px; }
  .inputWrap { padding: 0 16px 16px; }
}
```

- [ ] **步骤 3：验证构建**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误；`dist/` 生成。

- [ ] **步骤 4：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/src/App.tsx frontend/src/App.module.css && git commit -m "feat: 应用布局接线——hooks 组合与移动端抽屉

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 11：后端适配 web.py（TDD）

**文件：**
- 修改：`frontend/web.py:26-27`（常量）、`frontend/web.py:192-229`（create_app）、`frontend/web.py:276-278`（index 路由）
- 修改：`tests/test_web.py`

- [ ] **步骤 1：先改测试——首页断言替换为注入式用例**

在 `tests/test_web.py` 顶部 import 区增加：`from pathlib import Path`（已有 import 不含它）。

将 `test_create_app_serves_ui_health_and_streaming_chat` 函数体内的局部 `FakeService` 类提升为模块级（函数外），供新用例复用；删除该测试中的三行首页断言：

```python
    page = client.get("/")
    assert page.status_code == 200
    assert "AI 花音" in page.text
    assert 'id="record"' in page.text
    assert "/api/transcribe" in page.text
```

替换为两个独立用例（追加在文件末尾）：

```python
def test_index_serves_built_frontend_when_present(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from frontend.web import create_app

    page = tmp_path / "index.html"
    page.write_text('<title>AI 花音</title><div id="root"></div>', encoding="utf-8")
    client = TestClient(create_app(FakeService(), index_html=page))
    response = client.get("/")
    assert response.status_code == 200
    assert "AI 花音" in response.text
    assert '<div id="root">' in response.text


def test_index_missing_frontend_returns_build_hint():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from frontend.web import create_app

    client = TestClient(
        create_app(FakeService(), index_html=Path("/nonexistent/index.html"))
    )
    response = client.get("/")
    assert response.status_code == 503
    assert "npm run build" in response.text


def test_frontend_src_references_api_paths():
    src_root = Path(__file__).resolve().parents[1] / "frontend" / "src"
    if not src_root.is_dir():
        pytest.skip("frontend/src 尚未创建")
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(src_root.rglob("*.ts")) + sorted(src_root.rglob("*.tsx"))
    )
    for endpoint in ("/api/chat", "/api/transcribe", "/api/reset", "/healthz"):
        assert endpoint in sources, f"前端源码未引用 {endpoint}"
```

- [ ] **步骤 2：运行测试确认失败（红）**

运行：`cd /home/miku/dv_project/tts && pytest tests/test_web.py -x -q`
预期：FAIL —— `create_app() got an unexpected keyword argument 'index_html'`。

- [ ] **步骤 3：实现 web.py 改动**

常量区（`WEB_HTML` 附近）改为：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_HTML = PROJECT_ROOT / "frontend" / "dist" / "index.html"
WEB_ASSETS_DIR = PROJECT_ROOT / "frontend" / "dist" / "assets"
WEB_HINT_HTML = (
    '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
    "<title>AI 花音</title></head>"
    '<body style="background:#101416;color:#f1ece7;font:15px system-ui;'
    'display:grid;place-items:center;min-height:100vh">'
    "<p>Web 前端未构建，请先运行："
    "<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></p>"
    "</body></html>"
)
```

`create_app` 的 fastapi import 增加 StaticFiles：

```python
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
```

`create_app` 签名增加参数（在 `transcriber=None,` 之后）：

```python
    transcriber=None,
    max_sessions: int = 128,
    index_html: str | Path | None = None,
```

`app = FastAPI(...)` 之后增加静态资源挂载（仅构建产物存在时）：

```python
    app = FastAPI(title="AI 花音", version="0.1.0")
    if WEB_ASSETS_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_ASSETS_DIR), name="assets")
```

`index` 路由替换为：

```python
    @app.get("/", response_class=HTMLResponse)
    async def index():
        page = Path(index_html) if index_html is not None else WEB_HTML
        if not page.exists():
            return HTMLResponse(WEB_HINT_HTML, status_code=503)
        return page.read_text(encoding="utf-8")
```

- [ ] **步骤 4：运行测试确认通过（绿）**

运行：`cd /home/miku/dv_project/tts && pytest tests/test_web.py -q`
预期：PASS，全部用例通过（含两个新用例与 src 引用检查）。

- [ ] **步骤 5：Commit**

```bash
cd /home/miku/dv_project/tts && git add frontend/web.py tests/test_web.py && git commit -m "feat: web.py 服务 React 构建产物——index_html 注入与 /assets 挂载

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 12：清理旧前端与打包配置

**文件：**
- 删除：`frontend/web.html`
- 修改：`pyproject.toml:35-36`（package-data）
- 修改：`.gitignore`（若任务 1 未提交则补上）

- [ ] **步骤 1：删除 web.html 并更新 pyproject.toml**

删除 `frontend/web.html`：

```bash
cd /home/miku/dv_project/tts && git rm frontend/web.html
```

`pyproject.toml` 中删除：

```toml
[tool.setuptools.package-data]
frontend = ["web.html"]
```

（若该文件仅有此一个 package-data 块则整体删除。）

- [ ] **步骤 2：验证**

运行：`cd /home/miku/dv_project/tts && pytest -q`
预期：PASS（整个测试套件全绿，无引用 web.html 的测试残留）。
另运行：`grep -rn "web.html" --include="*.py" --include="*.toml" . | grep -v ".git/"` 预期：无输出。

- [ ] **步骤 3：Commit**

```bash
cd /home/miku/dv_project/tts && git add pyproject.toml && git commit -m "chore: 移除旧 web.html 与打包引用

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 13：README 更新

**文件：**
- 修改：`README.md`

- [ ] **步骤 1：更新三处 + 测试数**

1. 「功能特性」第 5 条改为：

```markdown
- **CLI 与 Web 双前端**：CLI 本地播放；Web（React + TypeScript）支持 SSE 流式文字 + WAV 音频、浏览器录音（faster-whisper 本地转写），暖夜深色界面
```

2. 「快速开始」第 4 节 Web 模式命令前增加构建步骤：

```markdown
# Web 模式（先构建前端，再启动服务；浏览器打开 http://127.0.0.1:8000/，健康检查 /healthz）
cd frontend && npm install && npm run build
cd .. && python -m frontend.web --host 127.0.0.1 --port 8000
```

3. 「项目结构」树中 `frontend/` 部分替换为：

```markdown
│   ├── frontend/
│   │   ├── src/                        # React + TypeScript 前端源码
│   │   ├── dist/                       # 构建产物（npm run build 生成，不入库）
│   │   ├── package.json / vite.config.ts / tsconfig.json
│   │   ├── cli.py                      # CLI 前端
│   │   └── web.py                      # FastAPI + SSE 对话服务（服务 dist/index.html）
```

4. 测试数量核对：运行 `pytest --collect-only -q | tail -1`，将 README 中的「153 个测试」改为实际数量。

- [ ] **步骤 2：验证**

运行：`cd /home/miku/dv_project/tts && pytest -q`
预期：PASS。

- [ ] **步骤 3：Commit**

```bash
cd /home/miku/dv_project/tts && git add README.md && git commit -m "docs: README 同步 Web 前端构建步骤与项目结构

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### 任务 14：集成验证与手动验收

**文件：** 无

- [ ] **步骤 1：全量自动化验证**

运行：`cd /home/miku/dv_project/tts/frontend && npm run build`
预期：tsc 零错误，dist 生成。

运行：`cd /home/miku/dv_project/tts && pytest -q`
预期：PASS。

- [ ] **步骤 2：手动验收（需要 TTS 服务 + API key，按规格 §7）**

启动 `python -m frontend.web` 后逐项勾选：

- [ ] 文字流式对话，逐句出现并自动播放语音
- [ ] 点击波形条暂停/继续，多句音频按序播放
- [ ] 录音转写 → 自动发送
- [ ] 清空对话
- [ ] 刷新页面后会话记忆保持
- [ ] 窄屏（≤768px）侧边栏折叠为抽屉，遮罩点击/ESC/按钮均可关闭
- [ ] 未构建 dist 时 `GET /` 返回 503 与「npm run build」提示

- [ ] **步骤 3：git 状态确认**

运行：`cd /home/miku/dv_project/tts && git status --short`
预期：仅 `frontend/dist/`、`frontend/node_modules/`（已被 gitignore 忽略）不出现；无其他未提交文件。

---

## 自检记录

- **规格覆盖度**：§2 决策（TS/不提交产物/暖夜深色/侧边栏/动效）→ 任务 1-10；§3.1 工程结构 → 任务 1-10；§3.2 后端适配（删 web.html、index_html 注入、/assets 挂载、503 提示页、部署说明）→ 任务 11-12；§3.3 测试（注入式双用例 + src 引用检查）→ 任务 11；§4 视觉系统（色板/角色卡/抽屉规范/动效）→ 任务 1、6-10；§5 数据流（类型/职责/会话持久化）→ 任务 2-5、10；§6 错误处理（mic/insecure/asr 态、busy、maxLength）→ 任务 4-8；§7 验证 → 任务 14；§8 工作流（README）→ 任务 13。
- **占位符扫描**：无 TODO/待定；每个步骤含完整代码与精确命令。
- **类型一致性**：`QueueItem{key,url}` 在 useAudioQueue（定义）与 useChat（enqueue 调用）、App（toggle 调用）三处一致；`Status` 11 态在 useChat 定义、StatusBadge 映射、App 组合处一致；`formatTime` 从 useAudioQueue 返回，经 App 传入 ChatMessage → AudioPlayer，签名一致；`create_app(index_html=...)` 在 web.py 与两处测试调用签名一致。
