#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOVITS_ROOT="${SOVITS_ROOT:-/home/mtr/tt/GPT-SoVITS}"
CONDA_ROOT="${CONDA_ROOT:-/home/mtr/miniconda3}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gptsovits

export CUDA_VISIBLE_DEVICES="${TTS_GPU:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SOVITS_ROOT:$SOVITS_ROOT/GPT_SoVITS:$SOVITS_ROOT/GPT_SoVITS/BigVGAN:$SOVITS_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"

exec python -u "$PROJECT_ROOT/scripts/test_huayin_tts.py" "$@"
