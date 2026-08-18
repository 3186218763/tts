# 运行时环境配置（推理，不含训练）

本文档面向要在新机器上把「AI 真白花音」对话系统跑起来的 agent 或运维。
目标：**一条命令完成环境配置，一条命令启动服务**。

## 硬件与系统假设

- Ubuntu 22.04（或含等价软件包的 Debian 系）；其它发行版需自行替换软件包名
- NVIDIA GPU ≥ 8GB 显存（实测 RTX 4070 Laptop 8GB 可用），驱动支持 CUDA 12.x
- 系统内存 ≥ 8GB；磁盘预留约 15GB（torch + 基础模型 + 交付权重）
- 已安装 `node/npm`（前端构建），`git`

## 一键配置

```bash
# 如 sudo 需要密码且无人值守，可 export SUDO_PASSWORD=... 供脚本使用
bash scripts/setup_runtime_env.sh
```

脚本幂等，重复执行只会补齐缺失部分。可覆盖的变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `SOVITS_ROOT` | `../GPT-SoVITS`（仓库同级目录） | GPT-SoVITS 代码位置 |
| `HF_MIRROR` | `https://hf-mirror.com` | 基础模型下载源（国内镜像） |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/cu128` | torch 下载源 |
| `TTS_PORT` | `9880` | TTS API 首选端口；被占/被拦自动回退 18080/19880 |

脚本步骤：

1. 系统依赖：`git-lfs python3.10-venv python3.10-dev cmake build-essential unzip curl`
2. 项目 venv：`uv sync --extra web --extra dev`（依赖 `uv.lock` 锁定版本，Python ≥3.11）
3. GPT-SoVITS：浅克隆官方仓库 → 独立 venv（Python 3.10）→ `torch+torchcodec`（cu128）→
   `torchaudio`（cu128，见下方坑 2）→ `requirements.txt`（跳过 `--no-binary=opencc`，opencc 装二进制轮子）
4. 推理基础模型（只下必需项，见下方清单）
5. `git lfs pull` 拉取交付权重（`model/huayin-gpt.ckpt` / `model/huayin-sovits.pth`）
6. 生成并修正 `configs/config.yaml`（TTS 端口 + 本机 `ref_audio_path` 绝对路径；LLM 的 provider/key/model 需自行填写）
7. `npm install && npm run build` 构建前端

## 启动与验证

```bash
bash scripts/run_local.sh           # 后台启动 TTS API + Web，日志在 logs/
bash scripts/run_local.sh --status  # 查看状态
bash scripts/run_local.sh --stop    # 停止

# 手动启动等价于：
$SOVITS_ROOT/.venv/bin/python scripts/run_huayin_api.py \
  --sovits-root $SOVITS_ROOT --python $SOVITS_ROOT/.venv/bin/python --port <TTS端口>
.venv/bin/python -m frontend.web --host 127.0.0.1 --port 8000
```

验证清单：

```bash
curl -s http://127.0.0.1:8000/healthz
# 期望 {"status":"ok",...,"llm_configured":true,"tts_available":true,"asr_available":true}

curl -s -X POST http://127.0.0.1:<TTS端口>/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，今天也要加油哦。","text_lang":"zh",
       "ref_audio_path":"<仓库>/model/huayin-ref.wav",
       "prompt_lang":"zh","prompt_text":"所以和朋友一起吃饭的话比较好哦",
       "media_type":"wav"}' -o /tmp/tts.wav
# 期望 HTTP 200 且 /tmp/tts.wav 是 RIFF/WAVE

curl -sN -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好","session_id":"smoke"}'
# 期望 SSE：sentence → audio → done，无 error 事件
```

LLM 最简配置示例（三选一，官方地址自动推导）：

```yaml
llm:
  provider: gemini       # openai / anthropic / gemini
  api_key: "your-api-key"
  model: "gemini-2.5-flash"
```

OpenAI 兼容的第三方网关再增加 `base_url`；旧配置中的 `protocol` 仍可继续使用。

## 推理所需基础模型清单

以下文件已够 v2Pro 推理；训练/分离（UVR5 等）不要装。

| 路径（相对 `$SOVITS_ROOT`） | 来源 |
|------|------|
| `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/*` | `hfl/chinese-roberta-wwm-ext-large` |
| `GPT_SoVITS/pretrained_models/chinese-hubert-base/*` | `TencentGameMate/chinese-hubert-base` |
| `GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` | `XXXXRT/GPT-SoVITS-Pretrained` |
| `GPT_SoVITS/text/G2PWModel/` | `XXXXRT/GPT-SoVITS-Pretrained/G2PWModel.zip` |
| `GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin` | fasttext 官方 |
| `~/nltk_data/corpora/cmudict`、`~/nltk_data/taggers/averaged_perceptron_tagger_eng` | NLTK 官方（英文 G2P 用） |

## 已知坑位（2026-08 实测）

1. **Git LFS 没装 → 模型只有 134 字节指针**。`apt install git-lfs && git lfs install && git lfs pull`。
2. **torchaudio 与 torch 的 CUDA 版本不匹配**。PyPI 默认 torchaudio 轮子按 CUDA 13 编译，
   cu128 的 torch 加载时报 `libcudart.so.13 not found`。必须从 cu128 索引重装：
   `pip install --force-reinstall --no-deps torchaudio --index-url <cu128索引>`。
3. **`opencc-python-reimplemented>=2.1` 不存在**（PyPI 最新 0.1.7）。`pyproject.toml`
   已改为 `>=0.1.7`；GPT-SoVITS 的 `requirements.txt` 里 `--no-binary=opencc` 会强制源码编译，
   配置脚本跳过该行改装二进制轮子。
4. **端口 9880~9882 可能被环境网络策略拦截**（绑定报 EADDRINUSE 但 `ss` 看不到占用）。
   脚本会自动探测并回退到 18080，同时写回 `config.yaml`。
5. **LLM 网关流末尾的空 `choices` chunk**。`opencode.ai/zen/go` 等网关会在流结束发
   `choices: []`，`dialogue/llm_client.py` 已加保护（跳过空 chunk），否则对话中段报
   `list index out of range` 且后续句子无音频。
6. **参考音固定**。当前决策：对话始终使用 `config.yaml` 的默认参考音
   （`model/huayin-ref.wav`），说话语气标签只做剥离、不切换参考音（`model/refs/` 暂不使用）。
   恢复切换逻辑：把 `dialogue/orchestrator.py` 与 `frontend/web.py` 中已移除的
   `style_bank.resolve` 代码块加回，并同步更新 `tests/test_orchestrator.py`。
7. **英文 G2P 依赖 NLTK 数据**。文本含英文（如“4K”“Q弹”）时缺 `cmudict` /
   `averaged_perceptron_tagger_eng` 会 400 报错，配置脚本已自动下载到 `~/nltk_data`。
8. **GPT-SoVITS 需要编译链**。`pyopenjtalk/jieba_fast` 源码编译需要
   `python3.x-dev` + `cmake` + `build-essential`。

## 目录约定

```
<工作区>/
├── tts/                  # 本仓库（项目根）
│   ├── .venv/            # 项目 venv（Python ≥3.11，uv 管理）
│   ├── model/            # 交付权重（LFS）+ 参考音库
│   ├── configs/config.yaml  # 本机配置（gitignore，脚本自动生成）
│   └── logs/             # 服务日志（gitignore）
└── GPT-SoVITS/           # 官方仓库浅克隆 + 独立 .venv（Python 3.10）
```
