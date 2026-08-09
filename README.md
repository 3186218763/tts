# AI 真白花音 TTS 交互对话系统

让用户与虚拟歌手 **真白花音（ましろ・はな）** 进行实时语音对话：输入文字（或语音），
云端 LLM 生成花音风格的台词，本地 GPT-SoVITS 以花音音色合成语音并播放。
项目同时包含从 B 站素材采集到 GPT-SoVITS 微调训练的完整数据管线。

## 功能特性

- **流式语音对话**：LLM 边生成、TTS 边合成、喇叭边播放，asyncio 三段流水线并行，首音延迟低
- **长对话记忆**：最近 `recent_turns` 轮保留原文，更早历史滚动压缩为 LLM 摘要，可支撑数百轮对话
- **自然语音切分**：按句末标点 / 逗号 / 空格切句，保护括号动作与引用，超长文本由 GPT-SoVITS `cut5` 兜底
- **CLI 与 Web 双前端**：CLI 本地播放；Web（React + TypeScript）支持 SSE 流式文字 + WAV 音频、浏览器录音（faster-whisper 本地转写），暖夜深色界面
- **训练数据管线**：白名单采集 → 人声分离 → 静音切分 → 歌声/能量过滤 → 声纹过滤 → ASR 转写 → 音文一致性校验，全阶段增量断点续跑
- **无头训练**：一条命令完成 GPT-SoVITS v2Pro 微调（文本特征 → Hubert → 语义 token → SoVITS → GPT）

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # 对话核心（CLI 播放需要 libsndfile / PortAudio）
pip install -e '.[web]'     # 额外安装 FastAPI + uvicorn + faster-whisper
```

### 2. 配置

```bash
cp configs/config.example.yaml configs/config.yaml
# 编辑 configs/config.yaml：填入 DeepSeek API key、模型与 GPT-SoVITS 地址
```

### 3. 启动 GPT-SoVITS TTS 服务

```bash
# 使用安装了 GPT-SoVITS 依赖的 Python 环境启动推理 API（默认 127.0.0.1:9880）
python scripts/run_huayin_api.py --python /path/to/gptsovits/bin/python
```

### 4. 对话

```bash
# CLI 模式（Ctrl+C 退出）
python -m frontend.cli

# Web 模式（先构建前端，再启动服务；浏览器打开 http://127.0.0.1:8000/，健康检查 /healthz）
cd frontend && npm install && npm run build
cd .. && python -m frontend.web --host 127.0.0.1 --port 8000
```

仅验证本地模型推理、不启动对话层：

```bash
python scripts/test_huayin_tts.py "你好，今天也要加油。"
```

完整部署说明（局域网访问、健康检查字段、录音功能）见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。

## 项目结构

```
├── config.py                    # 类型化配置加载（YAML → dataclass）
├── configs/config.example.yaml  # 配置模板（实际配置 config.yaml 被 gitignore）
├── dialogue/                    # 对话核心
│   ├── llm_client.py            # OpenAI 兼容 / Anthropic 原生双协议流式客户端 + 摘要接口
│   ├── tts_client.py            # GPT-SoVITS TTS API 客户端（统一接口，可换引擎）
│   ├── asr_client.py            # 本地 faster-whisper 转写（懒加载）
│   ├── conversation.py          # 对话历史：近期原文 + 滚动摘要 + 原子压缩
│   ├── memory.py                # CLI/Web 共用的记忆压缩与上下文组装
│   ├── sentence_streamer.py     # token 流 → 自然边界句子切分
│   ├── orchestrator.py          # LLM→切句→TTS→播放 的 asyncio 流水线
│   ├── audio_player.py          # sounddevice 播放（桌面端可选依赖）
│   ├── persona.py               # 花音人设 system prompt
│   └── speech_text.py           # 台词清洗：去 Markdown/括号动作/角色名前缀
├── frontend/
│   ├── src/                     # React + TypeScript 前端源码
│   ├── dist/                    # 构建产物（npm run build 生成，不入库）
│   ├── package.json / vite.config.ts / tsconfig.json
│   ├── cli.py                   # CLI 前端
│   └── web.py                   # FastAPI + SSE 对话服务（服务 dist/index.html）
├── scripts/                     # 数据管线 / 训练 / 运维脚本（见下表）
├── data/                        # 数据管线各阶段产物（见下）
├── model/                       # 训练好的模型权重（Git LFS）
└── tests/                       # pytest 测试（198 个，全部 mock，不依赖外部服务）
```

## 配置说明

| 配置节 | 关键项 | 说明 |
|--------|--------|------|
| `llm` | `api_key` / `base_url` / `model` / `protocol` | DeepSeek 兼容接口；`protocol` 支持 `openai`（默认）与 `anthropic` 原生协议；`temperature 0.8`、`max_tokens 400`、`frequency_penalty 0.15` 适合口语对话 |
| `tts` | `base_url` / `ref_audio_path` / `ref_text` | GPT-SoVITS 地址与参考音频；采样参数为可复现的保守默认（固定 `seed 42`） |
| `asr` | `model` / `device` / `compute_type` | 浏览器录音转写，默认 `small`，首次使用会下载模型 |
| `conversation` | `recent_turns` / `summary_trigger_turns` / `summary_trigger_chars` / `summary_max_chars` | 记忆分层阈值；旧配置 `max_turns` 仍作为 `recent_turns` 别名 |
| `streaming` | `min_sentence_chars` / `max_sentence_chars` | 切句边界，默认 4 / 50 字 |

## 对话机制

**记忆分层**：`Conversation` 保存最近 `recent_turns` 轮原文 + 较早历史的滚动摘要。
完整历史达到 `summary_trigger_turns` 轮或 `summary_trigger_chars` 字符时触发压缩：
先由 `plan_compaction()` 生成不可变压缩计划，摘要成功后 `apply_compaction()` 原子提交
（提交前校验历史未变，避免并发误删）。摘要失败则保留原文继续本轮，不影响主回复。
上下文顺序固定为：人设 system prompt → 记忆摘要（system）→ 近期原文 → 当前用户消息。

**句子切分**：`SentenceStreamer` 优先在句末标点切分；超过 `max_chars` 时在括号外的
逗号/顿号/空格处切；没有自然边界才切到硬上限，且不会从括号动作或引用中间硬切。

**流水线**：`Orchestrator.chat()` 中 LLM 流式输出 → 切句 → `tts_queue` 合成（失败只跳过
该句音频）→ `audio_queue` 顺序播放；LLM 异常时回滚当前用户消息。Web 端每个
`session_id` 有独立 `asyncio.Lock`，同一会话请求严格串行，SSE 事件类型：
`sentence` / `audio`（base64 WAV）/ `done` / `error` / `audio_error`。

## 数据管线

原始 WAV 放在 `data/wav/`，训练白名单在 `data/training_assets.txt`（逐行列出文件名）。
管线各阶段均支持增量断点续跑：

```
separate → slice → filter → speaker → asr → alignment → dataset
```

| 阶段 | 脚本 | 输入 → 输出 | 职责 |
|------|------|-------------|------|
| 采集 | `bili_download.py` / `batch_download.py` | B 站 → `data/wav/` | B 站 API 下载、搜索筛选、断点续传 |
| separate | `build_dataset.py --stage separate` / `run_parallel_separation.py` | `data/wav/` → `data/vocals/` | UVR5 BS-Roformer 人声分离，按白名单、长音频切块（默认 30 分钟）防内存溢出 |
| slice | `build_dataset.py --stage slice` | `data/vocals/` → `data/slices/` | 按 RMS 静音切分为短句，逐 vocal 增量 |
| filter | `build_dataset.py --stage filter` | `data/slices/` → `filter_results.json` | librosa `pyin` 音高特征：歌声（voiced_ratio / f0_cv / longest_voiced）、低能量、时长异常 |
| speaker | `filter_speakers.py` | 切片 + 声纹参考 → `speaker_results.json` | ECAPA-TDNN 声纹打分，剔除非花音人声 |
| asr | `build_dataset.py --stage asr` / `run_parallel_asr.py` | 保留切片 → `asr_results.json` | faster-whisper `large-v3` 增量转写 + 语种/置信度 |
| alignment | `verify_asr_alignment.py` | ASR 结果 → `alignment_results.json` | 强制语种二次转写，`SequenceMatcher` 校验音文一致性 |
| dataset | `build_dataset.py --stage dataset` | ASR + 对齐 → `data/dataset/` | 语种白名单（默认仅中日）、文本质量门、幻觉过滤，输出 `annotation.list` |

一次全量重建：

```bash
python scripts/build_dataset.py --stage all
python scripts/verify_asr_alignment.py --gpus 0 1 --min-similarity 0.45
python scripts/build_dataset.py --stage dataset --require-alignment
```

文本质量门（`dataset_text_quality.py`）以确定性规则检查短语循环、字符狂奔、韩文污染、
拉丁字符占比过高等 Whisper 幻觉模式。`run_dataset_supervisor.py` 可监控分离完成后
自动跑完后续阶段；`dataset_progress.py` 输出各阶段规模统计。

## 训练

```bash
python scripts/train_gpt_sovits.py \
  --exp-name huayin \
  --list-path data/dataset/annotation.list \
  --wav-dir data/dataset/audio \
  --version v2Pro \
  --gpus 0-1
```

脚本在 GPT-SoVITS 仓库环境下无头执行 WebUI 1A/1B/1C 全流程。当前交付模型：

- `model/huayin-e15.ckpt`：GPT 语义模型
- `model/huayin_e8_s2536.pth`：SoVITS 声学模型
- `model/huayin_ref.wav`：参考音频

## 脚本清单

| 脚本 | 职责 |
|------|------|
| `run_huayin_api.py` | 以交付权重启动 GPT-SoVITS `api_v2.py` 推理服务 |
| `test_huayin_tts.py` | 单条文本本地推理测试（支持 `--dry-run`） |
| `p0_verify.py` | 文字 → LLM → TTS → 播放 端到端链路验证 |
| `check_acceleration.py` | 检查当前环境可用的推理加速后端 |
| `setup_gpt_sovits.sh` / `run_train_tmux.sh` | GPT-SoVITS 环境准备 / tmux 训练启动 |
| `build_dataset.py` | 数据管线主入口（各阶段可单独运行） |
| `run_parallel_separation.py` / `run_parallel_asr.py` | 多 GPU 并行分离 / 转写 |
| `filter_speakers.py` | 花音声纹参考构建与逐切片打分 |
| `verify_asr_alignment.py` | 强制语种二次 ASR 音文一致性校验 |
| `dataset_text_quality.py` | 确定性文本质量门（幻觉转写检测） |
| `run_dataset_supervisor.py` / `finish_quality_pipeline.py` | 分离后自动完成后续管线 / 声纹后自动收尾 |
| `dataset_progress.py` | 打印管线各阶段进度统计 |
| `batch_download.py` / `bili_download.py` | B 站搜索下载（API 绕过反爬） |
| `sync_training_assets.py` | 审计通过的候选素材加入训练白名单 |
| `audit_*.py` | ASR 质量 / 音频资产 / 切片时长 / 声纹结果审计 |

## 测试

```bash
python -m pytest -q
```

198 个测试覆盖对话核心、切句、记忆压缩、流水线、Web SSE、配置加载与数据管线逻辑，
全部 mock 化，不依赖外部 API 或 GPU。

## 外部依赖说明

- 可选重依赖均为**懒加载**：`librosa`（filter 阶段）、`faster-whisper`（ASR / Web 录音）、
  `sounddevice` / `soundfile`（CLI 播放）、`audio_separator` 与 UVR5 权重（分离阶段，在
  GPT-SoVITS 环境中运行），不安装也不会阻断其他命令。
- `data/` 仅跟踪 `training_assets.txt`；模型权重、数据集、中间产物均由 `.gitignore` 排除。
