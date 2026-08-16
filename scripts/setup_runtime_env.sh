#!/usr/bin/env bash
# AI 真白花音 · 推理运行环境一键配置
#
# 覆盖：系统依赖 / 项目 venv / GPT-SoVITS 代码与 venv / 推理基础模型 /
#       模型权重 LFS / config.yaml / 前端构建。
# 幂等：已完成的步骤会自动跳过；下载均支持断点续传。
# 说明与排障见 docs/SETUP_RUNTIME.md。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOVITS_ROOT="$(realpath -m "${SOVITS_ROOT:-${PROJECT_ROOT}/../GPT-SoVITS}")"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

sudo_cmd() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif [ -n "${SUDO_PASSWORD:-}" ]; then
    echo "$SUDO_PASSWORD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

have() { command -v "$1" >/dev/null 2>&1; }

python_bindable() {
  python3 - "$1" <<'PY_BIND_EOF'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
    print("ok")
except OSError:
    print("busy")
finally:
    s.close()
PY_BIND_EOF
}

pick_tts_port() {
  for candidate in "${TTS_PORT:-9880}" 18080 19880; do
    if [ "$(python_bindable "$candidate")" = "ok" ]; then
      echo "$candidate"
      return 0
    fi
    log "端口 $candidate 不可用（被占用或环境策略拦截），换下一个候选"
  done
  fail "找不到可用的 TTS 端口"
}

# ---------------------------------------------------------------- system deps
log "1/7 系统依赖（git-lfs / python3.10 venv+dev / 编译链）"
sudo_cmd apt-get update -qq
sudo_cmd apt-get install -y -qq \
  git-lfs python3.10-venv python3.10-dev cmake build-essential unzip curl

# ------------------------------------------------------------------ project venv
log "2/7 项目 venv（uv + 锁定依赖）"
if ! have uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$PROJECT_ROOT"
uv sync --extra web --extra dev
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# ------------------------------------------------- GPT-SoVITS code + venv
log "3/7 GPT-SoVITS 代码与 venv"
mkdir -p "$(dirname "$SOVITS_ROOT")"
if [ ! -d "$SOVITS_ROOT/.git" ]; then
  git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git "$SOVITS_ROOT"
fi
SOVITS_PYTHON="${SOVITS_PYTHON:-python3.10}"
if ! have "$SOVITS_PYTHON"; then SOVITS_PYTHON=python3; fi
if [ ! -x "$SOVITS_ROOT/.venv/bin/python" ]; then
  "$SOVITS_PYTHON" -m venv "$SOVITS_ROOT/.venv"
fi
SOVITS_BIN="$SOVITS_ROOT/.venv/bin/python"
"$SOVITS_BIN" -m pip install -q -U pip
if ! "$SOVITS_BIN" -c 'import torch, torchaudio, torchcodec' 2>/dev/null; then
  cd "$SOVITS_ROOT"
  "$SOVITS_BIN" -m pip install -q torch torchcodec --index-url "$PYTORCH_INDEX"
  "$SOVITS_BIN" -m pip install -q --force-reinstall --no-deps torchaudio --index-url "$PYTORCH_INDEX"
fi
if ! "$SOVITS_BIN" -c 'import funasr, transformers' 2>/dev/null; then
  cd "$SOVITS_ROOT"
  grep -v '^--no-binary' requirements.txt > /tmp/gsv-requirements.txt
  "$SOVITS_BIN" -m pip install -q -r /tmp/gsv-requirements.txt
  "$SOVITS_BIN" -m pip install -q opencc
fi

# ------------------------------------------------------------------ base models
log "4/7 GPT-SoVITS 推理基础模型（BERT/HuBERT/G2PW/声纹/语种检测）"
PM="$SOVITS_ROOT/GPT_SoVITS/pretrained_models"
mkdir -p "$PM/chinese-roberta-wwm-ext-large" "$PM/chinese-hubert-base" \
         "$PM/sv" "$PM/fast_langdetect" "$SOVITS_ROOT/GPT_SoVITS/text"
fetch() { # fetch <url> <dest>
  if [ -s "$2" ]; then log "已存在，跳过: $2"; return 0; fi
  mkdir -p "$(dirname "$2")"
  curl -sL --retry 3 -C - -o "$2" "$1"
}
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt; do
  fetch "$HF_MIRROR/hfl/chinese-roberta-wwm-ext-large/resolve/main/$f" "$PM/chinese-roberta-wwm-ext-large/$f"
done
fetch "$HF_MIRROR/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin" "$PM/chinese-roberta-wwm-ext-large/pytorch_model.bin"
for f in config.json preprocessor_config.json; do
  fetch "$HF_MIRROR/TencentGameMate/chinese-hubert-base/resolve/main/$f" "$PM/chinese-hubert-base/$f"
done
fetch "$HF_MIRROR/TencentGameMate/chinese-hubert-base/resolve/main/pytorch_model.bin" "$PM/chinese-hubert-base/pytorch_model.bin"
fetch "$HF_MIRROR/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt" "$PM/sv/pretrained_eres2netv2w24s4ep4.ckpt"
fetch "$HF_MIRROR/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip" /tmp/G2PWModel.zip
if [ ! -d "$SOVITS_ROOT/GPT_SoVITS/text/G2PWModel" ]; then
  unzip -q -o /tmp/G2PWModel.zip -d "$SOVITS_ROOT/GPT_SoVITS/text/"
fi
fetch https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin "$PM/fast_langdetect/lid.176.bin"

# NLTK（英文 G2P 需要 cmudict + 词性标注）
mkdir -p "$HOME/nltk_data/corpora" "$HOME/nltk_data/taggers"
fetch https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip /tmp/cmudict.zip
[ -d "$HOME/nltk_data/corpora/cmudict" ] || (cd "$HOME/nltk_data/corpora" && unzip -q -o /tmp/cmudict.zip)
fetch https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip /tmp/tagger.zip
[ -d "$HOME/nltk_data/taggers/averaged_perceptron_tagger_eng" ] || (cd "$HOME/nltk_data/taggers" && unzip -q -o /tmp/tagger.zip)

# --------------------------------------------------------------- model weights
log "5/7 交付模型权重（Git LFS）"
cd "$PROJECT_ROOT"
git lfs install --local
git lfs pull

# ---------------------------------------------------------------------- config
log "6/7 configs/config.yaml"
CONFIG="$PROJECT_ROOT/configs/config.yaml"
[ -f "$CONFIG" ] || cp "$PROJECT_ROOT/configs/config.example.yaml" "$CONFIG"
TTS_PORT="$(pick_tts_port)"
sed -i -E "s|^(  base_url: )\"http://127\.0\.0\.1:[0-9]+\"|\1\"http://127.0.0.1:${TTS_PORT}\"|" "$CONFIG"
sed -i -E "s|^(  ref_audio_path: ).*|\1\"$PROJECT_ROOT/model/huayin-ref.wav\"|" "$CONFIG"
log "TTS 端口: $TTS_PORT；请确认 $CONFIG 中 llm.api_key 已填写"

# -------------------------------------------------------------------- frontend
log "7/7 前端构建"
cd "$PROJECT_ROOT/frontend"
npm install --no-audit --no-fund
npm run build

# -------------------------------------------------------------------- summary
cat <<SETUP_SUMMARY_EOF

[setup] 环境配置完成
  - 项目 venv:      $PROJECT_ROOT/.venv
  - GPT-SoVITS:     $SOVITS_ROOT（venv: $SOVITS_ROOT/.venv）
  - TTS API 端口:   $TTS_PORT
  - Web 端口:       8000

  启动服务:
    bash $PROJECT_ROOT/scripts/run_local.sh

  手动启动:
    $SOVITS_ROOT/.venv/bin/python $PROJECT_ROOT/scripts/run_huayin_api.py \
      --sovits-root $SOVITS_ROOT --python $SOVITS_ROOT/.venv/bin/python --port $TTS_PORT
    $PROJECT_ROOT/.venv/bin/python -m frontend.web --host 127.0.0.1 --port 8000
SETUP_SUMMARY_EOF
