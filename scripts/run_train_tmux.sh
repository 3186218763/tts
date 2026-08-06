#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOVITS_ROOT="${SOVITS_ROOT:-/home/mtr/tt/GPT-SoVITS}"
CONDA_ROOT="${CONDA_ROOT:-/home/mtr/miniconda3}"
SESSION="${TMUX_SESSION:-huayin-v2pro-train}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/data/dataset}"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/train_huayin_v2pro.log"
SYSTEM_LIBFFI="/lib/x86_64-linux-gnu/libffi.so.7"

run_training() {
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate gptsovits

  mkdir -p "$LOG_DIR"
  ulimit -n 65535 || true
  export PYTHONUNBUFFERED=1
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0,1
  export PYTHONPATH="$SOVITS_ROOT:$SOVITS_ROOT/GPT_SoVITS:$SOVITS_ROOT/GPT_SoVITS/BigVGAN:$SOVITS_ROOT/tools:$SOVITS_ROOT/tools/asr:$SOVITS_ROOT/tools/uvr5${PYTHONPATH:+:$PYTHONPATH}"
  export FORMAT_WORKER_THREADS="${FORMAT_WORKER_THREADS:-3}"
  export OMP_NUM_THREADS="$FORMAT_WORKER_THREADS"
  export MKL_NUM_THREADS="$FORMAT_WORKER_THREADS"
  export OPENBLAS_NUM_THREADS="$FORMAT_WORKER_THREADS"
  export NUMEXPR_NUM_THREADS="$FORMAT_WORKER_THREADS"
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_IB_DISABLE=1

  local format_args=()
  if [[ "${SKIP_FORMAT:-0}" == "1" ]]; then
    format_args+=(--skip-format)
    echo "skip dataset formatting (SKIP_FORMAT=1)" | tee -a "$LOG_FILE"
  fi

  [[ -f "$DATASET_DIR/annotation.list" ]] || {
    echo "annotation list missing: $DATASET_DIR/annotation.list" | tee -a "$LOG_FILE"
    return 2
  }
  [[ -d "$DATASET_DIR/audio" ]] || {
    echo "audio directory missing: $DATASET_DIR/audio" | tee -a "$LOG_FILE"
    return 2
  }
  [[ -f "$SYSTEM_LIBFFI" ]] || {
    echo "system libffi missing: $SYSTEM_LIBFFI" | tee -a "$LOG_FILE"
    return 2
  }

  local sample_rel
  local sample_wav
  sample_rel="$(awk -F'|' 'NF { print $1; exit }' "$DATASET_DIR/annotation.list")"
  sample_wav="$DATASET_DIR/audio/$(basename "$sample_rel")"
  if ! LD_PRELOAD="$SYSTEM_LIBFFI" python - "$sample_wav" <<'PY'
import sys
import torchaudio
import torchcodec

wav, sample_rate = torchaudio.load(sys.argv[1])
assert sample_rate == 32000
assert wav.numel() > 0
print(f"torchcodec preflight passed: {torchcodec.__version__}, {tuple(wav.shape)} @ {sample_rate} Hz")
PY
  then
    echo "torchcodec preflight failed; v2Pro speaker features cannot be generated" | tee -a "$LOG_FILE"
    return 2
  fi

  cd "$SOVITS_ROOT"
  echo "==== train start $(date) session=$SESSION ====" | tee -a "$LOG_FILE"
  # S1 attention scales sharply with sequence length; 8 is stable on 24 GiB cards.
  python -u "$PROJECT_ROOT/scripts/train_gpt_sovits.py" \
    --sovits-root "$SOVITS_ROOT" \
    --exp-name huayin \
    --list-path "$DATASET_DIR/annotation.list" \
    --wav-dir "$DATASET_DIR/audio" \
    --version v2Pro \
    --gpus 0-1 \
    --format-workers-per-gpu "${FORMAT_WORKERS_PER_GPU:-6}" \
    --batch-size-s2 "${BATCH_SIZE_S2:-12}" \
    --batch-size-s1 "${BATCH_SIZE_S1:-8}" \
    --epochs-s2 "${EPOCHS_S2:-8}" \
    --epochs-s1 "${EPOCHS_S1:-15}" \
    --save-every-s2 "${SAVE_EVERY_S2:-4}" \
    --save-every-s1 "${SAVE_EVERY_S1:-1}" \
    "${format_args[@]}" \
    2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  echo "==== train end exit=$status $(date) ====" | tee -a "$LOG_FILE"
  return "$status"
}

start_session() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
  fi
  # tmux servers keep their own environment; pass launch-time overrides into the pane explicitly.
  local env_prefix="env"
  local env_name
  local env_value
  for env_name in \
    TMUX_SESSION SOVITS_ROOT CONDA_ROOT DATASET_DIR SKIP_FORMAT \
    FORMAT_WORKERS_PER_GPU FORMAT_WORKER_THREADS BATCH_SIZE_S2 BATCH_SIZE_S1 \
    EPOCHS_S2 EPOCHS_S1 SAVE_EVERY_S2 SAVE_EVERY_S1; do
    if [[ -v "$env_name" ]]; then
      printf -v env_value '%q' "${!env_name}"
      env_prefix+=" ${env_name}=${env_value}"
    fi
  done
  tmux new-session -d -s "$SESSION" "$env_prefix bash '$PROJECT_ROOT/scripts/run_train_tmux.sh' --inside-tmux"
  tmux new-window -d -t "$SESSION:" -n monitor "watch -n 2 nvidia-smi"
  tmux select-window -t "$SESSION:0"
  echo "started tmux session: $SESSION"
  echo "attach: tmux attach -t $SESSION"
  echo "log: $LOG_FILE"
}

case "${1:-start}" in
  start)
    start_session
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux kill-session -t "$SESSION"
      echo "stopped tmux session: $SESSION"
    else
      echo "tmux session not found: $SESSION"
    fi
    ;;
  attach)
    exec tmux attach -t "$SESSION"
    ;;
  status)
    tmux list-panes -t "$SESSION" -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command}'
    ;;
  --inside-tmux)
    set +e
    run_training
    status=$?
    echo "training process exited with code $status"
    exec bash
    ;;
  *)
    echo "usage: $0 {start|stop|attach|status}" >&2
    exit 2
    ;;
esac
