# Web 前端重设计规格：React + TypeScript 暖夜深色版

日期：2026-08-09
状态：已批准（用户确认第 1–5 节设计）

## 1. 背景与目标

现状 `frontend/web.html` 是单文件原生 JS 页面（内联 CSS + 内联脚本），由 FastAPI 直接读取返回。功能完整（SSE 流式对话、录音转写、会话管理）但视觉朴素、代码不可维护。

本次目标：

1. **美观优化**：以「暖夜深色」方向重做视觉（深炭底 + 桃金暖光，深夜直播氛围），贴合花音「元气、温柔」的纪念向人设。
2. **技术升级**：前端全部改用 **TypeScript**（strict 模式），不写任何 `.js` 源码；采用 React + Vite 工程化结构。
3. **后端 API 零改动**：仅 `GET /` 的静态文件来源改变；`/api/chat`、`/api/transcribe`、`/api/reset`、`/healthz` 全部不变。

## 2. 已确认决策

| 决策点 | 选择 |
|---|---|
| 框架 | React + TypeScript（Vite 构建） |
| 构建产物 | **不提交 git**（`dist/` 进 `.gitignore`），README 增加 npm 前置步骤 |
| 视觉风格 | A · 暖夜深色（深色底 + 桃金暖光） |
| 布局 | 单页 + 角色侧边栏（≤768px 折叠为抽屉） |
| 动效 | 精致动效（CSS 过渡 + Web Audio 波形可视化） |
| 后端 | `web.py` 服务 `frontend/dist/index.html`，缺失时返回提示页 |

## 3. 技术架构

### 3.1 前端工程

```
frontend/
├── package.json / vite.config.ts / tsconfig.json
├── index.html                # 入口模板（标题、挂载点、字体）
└── src/
    ├── main.tsx              # 挂载 React
    ├── App.tsx               # 布局：Sidebar + 聊天主区
    ├── api/client.ts         # 类型化 fetch：SSE 解析、transcribe、reset、healthz
    ├── types.ts              # ChatEvent / ChatMessage / HealthStatus 类型
    ├── hooks/
    │   ├── useChat.ts        # 消息列表、发送、SSE 消费、busy 状态
    │   ├── useAudioQueue.ts  # 音频顺序播放 + AnalyserNode 波形数据
    │   └── useRecorder.ts    # MediaRecorder 录音 → 上传转写 → 自动发送
    ├── components/
    │   ├── Sidebar.tsx       # 角色卡（头像/人设/状态/清空按钮）
    │   ├── ChatMessage.tsx   # 气泡 + 内嵌 AudioPlayer
    │   ├── AudioPlayer.tsx   # 播放按钮 + 波形条 + 时长
    │   ├── ChatInput.tsx     # 录音/输入框/发送
    │   └── StatusBadge.tsx   # 顶部状态徽标
    └── styles/tokens.css     # CSS 变量设计令牌
```

工程约束：

- `tsconfig.json` 开启 `strict: true`，源码不得出现 `any`；所有文件为 `.ts` / `.tsx`（Vite 配置文件 `vite.config.ts` 亦为 TS）。
- `vite.config.ts` 配置 dev proxy：`/api`、`/healthz` → `http://127.0.0.1:8000`。
- 不引入状态管理库（zustand 等）——单页应用用 hooks 组合足够（YAGNI）。
- 样式：`tokens.css`（设计令牌）+ 组件级 CSS Modules。

### 3.2 后端适配（`frontend/web.py`）

- `WEB_HTML` 常量改为指向 `PROJECT_ROOT / "frontend" / "dist" / "index.html"`。
- **静态资源挂载**：Vite 构建产物按 `/assets/*` 引用，需新增 `app.mount("/assets", StaticFiles(directory=frontend/dist/assets))`；仅当 `dist/assets` 目录存在时才挂载（避免未构建时启动报错）。
- `GET /` 时若 `dist/index.html` 不存在，返回 **503 状态 + 含提示文案的简单 HTML 页**（「Web 前端未构建，请先 `cd frontend && npm install && npm run build`」）。
- 其余路由与逻辑不变。

### 3.3 测试更新（`tests/test_web.py`）

仓库惯例是 `pip install -e . && pytest` 即可全绿（无 Node 环境），因此首页断言改为**双分支**：

- `dist/index.html` 存在 → `GET /` 为 200，HTML 含标题「AI 花音」与挂载根节点（如 `<div id="root">`）；
- `dist/index.html` 缺失 → `GET /` 为 503，HTML 含提示文案「npm run build」。
- `/api/transcribe` 等 API 路径出现在 `frontend/src/` 源码中（确保前端确实引用这些端点；源码随 git 提交，不依赖构建）。
- `/api/chat`、`/api/transcribe`、`/api/reset`、`/healthz` 的 API 行为测试**全部不变**。

## 4. 视觉设计系统

### 4.1 色板（tokens.css）

| 令牌 | 值 | 用途 |
|---|---|---|
| `--bg` | `#0f1315` | 页面基底 |
| `--bg-glow` | `radial-gradient(55% 45% at 78% -5%, rgba(255,160,110,.16), transparent 60%)` 等 | 右上桃金光晕 + 左下淡光 |
| `--panel` | `rgba(255,255,255,.03)` | 侧边栏/输入栏玻璃面板 |
| `--line` | `rgba(255,255,255,.07–.12)` | 边框 |
| `--text` | `#f1ece7` | 主文本 |
| `--muted` | `#9eaeab` | 次要文本 |
| `--accent` | `#f2b6a0` → `#ff917d`（渐变） | 主强调色（桃金） |
| `--ai-bubble` | 渐变 `#2b2430 → #26212c`，边框 `rgba(255,255,255,.09)` | AI 气泡 |
| `--user-bubble` | 渐变 `#1e3a3d → #1b3437`，边框 `rgba(120,190,185,.25)` | 用户气泡 |
| `--ok` | `#7fcf9a` | 在线/健康 |
| `--danger` | `#ec665d` | 录音中/错误 |

圆角：气泡 14px（近边 5px）、输入栏 16px、侧边栏内容 10px、徽标 pill 999px。
阴影：面板 `0 8px 26px rgba(0,0,0,.35)`、气泡 `0 4px 14px rgba(0,0,0,.25)`。
字体：`system-ui` 栈（与现状一致，不引外部字体）。

### 4.2 侧边栏角色卡（`Sidebar.tsx`）

自上而下：

1. **「菜」字徽标**：72px 圆形，中心深灰渐变 + 桃色「菜」字；外圈 `conic-gradient(#f2b6a0 → #ff917d → #8a5a4e)` 细光环 + `drop-shadow` 光晕；底部椭圆柔光（纯 CSS，无需图片素材）。
2. **昵称**：真白花音（17px，650 字重）；副题「AI 复刻 · 纪念向」（11px muted）。
3. **状态 pill**：「在线 · 深夜电台」，绿色呼吸圆点 + `box-shadow` 光晕。
4. **人设简介**（左边框强调线）：「元气、温柔、笨拙但倔强。2020 年起在 B 站开播，2026-05-01 毕业。这里是粉丝为她搭建的 AI 纪念亭——她已不在，但声音还在。」
5. **口号**：「每天都和白菜在一起」（斜体，桃色淡）。
6. **底部**：「↻ 清空对话」按钮（hover 桃色描边）+ 小字「AI 复刻纪念项目 · 非本人」。

### 4.3 聊天主区

- 顶栏：左「AI 花音」品牌字，右 `StatusBadge`（状态徽标：在线 / 生成中 / 录音中 / 识别中 / 请配置 DeepSeek API key / TTS 服务未启动 / 在线（语音不可用） / 连接异常）。
- 消息列表：居中 `max-width 760px` 消息列；AI 气泡暖紫灰渐变靠左、用户气泡深青渐变靠右，气泡带微妙渐变、边框与阴影。
- 音频条（`AudioPlayer.tsx`）：桃金圆播放钮 + 24 条波形竖条 + 时长（`0:00 / 0:03`）；播放中波形由 `AnalyserNode` 实时驱动并脉冲跳动，点击切换播放/暂停；同一时刻全局最多一个音频在播（顺序队列）。
- 输入栏（`ChatInput.tsx`）：圆角 16px 玻璃面板；左录音按钮（红点，录音中变方 + 呼吸脉冲，`MediaRecorder` 30 秒上限）、中 `textarea`（占位「输入消息...（Enter 发送，Shift+Enter 换行）」，`maxlength=2000`）、右桃金渐变发送钮。

### 4.4 动效清单

| 动效 | 实现 | 参数 |
|---|---|---|
| 消息入场 | CSS `fadeIn + translateY(8px)` | 160ms ease-out |
| 波形跳动 | `AnalyserNode` + rAF（真实频谱）或 CSS 动画（未播放状态预览） | 24 条 |
| 录音脉冲 | CSS 呼吸动画 + 红点→方角 | 1.2s |
| 背景光晕呼吸 | 背景层透明度循环 | 8s ease-in-out |
| 悬停/按下过渡 | CSS transition | 160–220ms |
| 播放中波形高亮 | 当前 `audio.url` 匹配的条目加 `playing` 类 | — |

## 5. 数据流与状态

### 5.1 类型（`types.ts`）

```ts
type ChatEvent =
  | { type: "sentence"; text: string }
  | { type: "audio"; audio: string }        // base64 WAV
  | { type: "audio_error"; message: string }
  | { type: "error"; message: string }
  | { type: "done" };

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;                             // 随 sentence 事件累积
  audio: { url: string; error?: string }[]; // data: URL 或合成失败标记
}
```

### 5.2 职责划分

- **`useChat`**：`messages` 列表；`send()` 追加用户气泡 → 创建空 AI 气泡 → `fetch /api/chat`（POST JSON `{message, session_id}`）→ 按 `\n\n` 分块解析 `data: ` 行 → 依事件类型填充气泡 / 入音频队列 / 抛错；`busy` 锁；`reset()` 调 `/api/reset` 并清空列表；启动时 `GET /healthz` 决定初始徽标与录音按钮可用性。
- **`useAudioQueue`**：单例队列，`enqueue(url)` → 顺序播放（`ended` 触发下一个）；`HTMLAudioElement` + `AudioContext.createMediaElementSource` → `analyser.getByteFrequencyData` → rAF 输出 24 条波形高度；暴露 `currentUrl`、`isPlaying`、`toggle(url)`。
- **`useRecorder`**：`getUserMedia({audio: {channelCount:1, echoCancellation:true, noiseSuppression:true}})` → 优先 `audio/webm;codecs=opus` → 录音 30s 上限 → `POST /api/transcribe`（`Content-Type` 为实际媒质类型）→ 文本填入输入框并自动发送。权限拒绝 / ASR 不可用均有提示。

### 5.3 会话与持久化

- `session_id`：`localStorage["huayin-session-id"]`（UUID），现状不变。
- 刷新后消息列表清空但会话记忆保留（后端侧），与现状行为一致。

## 6. 错误处理

| 场景 | 处理 |
|---|---|
| 后端不可用 / 网络断 | 当前 AI 气泡显示「请求失败：…」，徽标「连接异常」 |
| SSE `error` 事件 | 同气泡终止并显示原因 |
| `audio_error` 事件 | 文字保留，音频条替换为「语音合成失败」小字 |
| 麦克风权限被拒 | 徽标提示「无法访问麦克风」 |
| ASR 未配置 | 录音按钮禁用 + tooltip「语音模型未安装」 |
| 生成中（busy） | 输入与录音禁用 |
| 输入超 2000 字 | `maxlength` 硬限制 + 字数提示 |

## 7. 测试与验证

1. `cd frontend && npm run build` — `tsc --noEmit`（strict）零错误 + vite 产出 `dist/`。
2. `pytest tests/test_web.py` — 更新后全绿（API 测试不变）。
3. 手动验收清单：
   - [ ] 文字流式对话，逐句出现并自动播放语音；
   - [ ] 点击波形条暂停/继续，多句音频按序播放；
   - [ ] 录音转写 → 自动发送；
   - [ ] 清空对话；
   - [ ] 刷新页面后会话记忆保持；
   - [ ] 窄屏（≤768px）侧边栏折叠为抽屉；
   - [ ] 未构建 dist 时 `GET /` 返回明确提示。

## 8. 开发与部署工作流

```bash
# 开发（热更新，proxy 转发 /api、/healthz 到 8000）
cd frontend && npm install && npm run dev

# 使用（构建后照旧 pip 即用）
cd frontend && npm run build
pip install -e '.[web]'
python -m frontend.web --host 127.0.0.1 --port 8000
```

README「功能特性」「快速开始」「项目结构」三处同步更新（Web 部分加 npm 前置步骤，目录结构换为新工程布局）。

## 9. 不做的事（YAGNI）

- 不做多页面/路由（聊天 + 关于 + 设置）；
- 不引入状态管理库、UI 组件库；
- 不提交 `dist/` 构建产物；
- 不做粒子背景、鼠标视差等炫技动效；
- 后端 API 与对话逻辑不做任何改动。
