#!/usr/bin/env bash
# 高精度子集训练（tmux）：data/dataset_precision → huayin-precision
#
# 参数来源（单一真相）: configs/huayin_precision.yaml
# 设计见 docs/TRAINING_PRECISION.md
#
# 用法:
#   bash scripts/run_precision_train_tmux.sh start
#   bash scripts/run_precision_train_tmux.sh stop
#   bash scripts/run_precision_train_tmux.sh attach
#   bash scripts/run_precision_train_tmux.sh status
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_YAML="${RECIPE_YAML:-$PROJECT_ROOT/configs/huayin_precision.yaml}"
export CONDA_ROOT="${CONDA_ROOT:-/home/mtr/miniconda3}"

# Load locked recipe into env (env already set by caller wins over recipe).
if [[ -f "$RECIPE_YAML" ]]; then
  # shellcheck disable=SC1090
  eval "$(
    python3 "$PROJECT_ROOT/scripts/huayin_precision_recipe.py" \
      --recipe "$RECIPE_YAML" --export-env
  )"
else
  echo "WARN: recipe missing $RECIPE_YAML; using script fallbacks" >&2
  export SOVITS_ROOT="${SOVITS_ROOT:-/home/mtr/tt/GPT-SoVITS}"
  export TMUX_SESSION="${TMUX_SESSION:-huayin-precision-train}"
  export DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/data/dataset_precision}"
  export EXP_NAME="${EXP_NAME:-huayin-precision}"
  export BATCH_SIZE_S2="${BATCH_SIZE_S2:-8}"
  export BATCH_SIZE_S1="${BATCH_SIZE_S1:-8}"
  export EPOCHS_S2="${EPOCHS_S2:-12}"
  export EPOCHS_S1="${EPOCHS_S1:-25}"
  export SAVE_EVERY_S2="${SAVE_EVERY_S2:-2}"
  export SAVE_EVERY_S1="${SAVE_EVERY_S1:-1}"
  export FORMAT_WORKERS_PER_GPU="${FORMAT_WORKERS_PER_GPU:-4}"
  export FORMAT_WORKER_THREADS="${FORMAT_WORKER_THREADS:-2}"
  export LR_S1="${LR_S1:-0.005}"
  export S1_WARMUP_STEPS="${S1_WARMUP_STEPS:-80}"
  export LR_DECAY_STEPS="${LR_DECAY_STEPS:-450}"
  export S1_DEV_FRAC="${S1_DEV_FRAC:-0.10}"
  export S1_VAL_BATCHES="${S1_VAL_BATCHES:-40}"
  export S1_DROPOUT="${S1_DROPOUT:-0.1}"
  export S2_DROPOUT="${S2_DROPOUT:-0.1}"
  export S1_EARLY_STOP_PATIENCE="${S1_EARLY_STOP_PATIENCE:-5}"
  export S1_EARLY_STOP_MIN_DELTA="${S1_EARLY_STOP_MIN_DELTA:-0.001}"
  export SKIP_FORMAT="${SKIP_FORMAT:-1}"
  export LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/train_huayin_precision.log}"
fi

# Fresh base models (do not resume overfit S1) unless FORCE_* set
if [[ -n "${FORCE_INIT_S1:-}" ]]; then
  export INIT_S1="$FORCE_INIT_S1"
else
  unset INIT_S1 2>/dev/null || true
fi
if [[ -n "${FORCE_INIT_S2G:-}" ]]; then
  export INIT_S2G="$FORCE_INIT_S2G"
else
  unset INIT_S2G 2>/dev/null || true
fi

export SKIP_S2="${SKIP_S2:-0}"
export DRY_RUN="${DRY_RUN:-0}"
# Allow explicit SKIP_FORMAT override after recipe load
export SKIP_FORMAT="${SKIP_FORMAT:-1}"

LOG_DIR="$PROJECT_ROOT/logs"
export LOG_FILE="${LOG_FILE:-$LOG_DIR/train_huayin_precision.log}"
mkdir -p "$LOG_DIR"

preflight() {
  [[ -f "$DATASET_DIR/annotation.list" ]] || {
    echo "missing $DATASET_DIR/annotation.list" >&2
    exit 2
  }
  [[ -d "$DATASET_DIR/audio" ]] || {
    echo "missing $DATASET_DIR/audio" >&2
    exit 2
  }
  local n
  n="$(grep -c . "$DATASET_DIR/annotation.list" || true)"
  echo "recipe: $RECIPE_YAML"
  echo "dataset: $DATASET_DIR ($n lines) exp=$EXP_NAME session=$TMUX_SESSION"
  echo "S2: epochs=$EPOCHS_S2 bs=$BATCH_SIZE_S2 save_every=$SAVE_EVERY_S2 dropout=$S2_DROPOUT"
  echo "S1: epochs=$EPOCHS_S1 bs=$BATCH_SIZE_S1 lr=$LR_S1 warmup=$S1_WARMUP_STEPS decay=$LR_DECAY_STEPS dev=$S1_DEV_FRAC dropout=$S1_DROPOUT early_stop_patience=$S1_EARLY_STOP_PATIENCE"
  echo "SKIP_FORMAT=$SKIP_FORMAT SKIP_S2=$SKIP_S2"
  # Drop orphaned FunASR/Whisper spawn workers that may hold VRAM.
  # Match only real python workers; never kill self or the trainer shell.
  local pids=""
  local pid cmd
  while read -r pid cmd; do
    [[ -z "$pid" || "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    case "$cmd" in
      *run_precision_train*|*run_train_tmux*|*train_gpt_sovits*) continue ;;
    esac
    if [[ "$cmd" == *gptsovits* && "$cmd" == *multiprocessing.spawn*spawn_main* ]]; then
      pids+="$pid "
    fi
  done < <(ps -eo pid=,args=)
  if [[ -n "${pids// /}" ]]; then
    echo "killing orphan gptsovits spawn workers: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
  fi
}

case "${1:-start}" in
  start)
    preflight
    {
      echo "==== precision train launch $(date) ===="
      echo "DATASET_DIR=$DATASET_DIR EXP_NAME=$EXP_NAME"
      echo "EPOCHS_S2=$EPOCHS_S2 EPOCHS_S1=$EPOCHS_S1 LR_S1=$LR_S1 WARMUP=$S1_WARMUP_STEPS DECAY=$LR_DECAY_STEPS DEV=$S1_DEV_FRAC"
    } | tee -a "$LOG_FILE"
    # Reuse generic trainer (passes env into tmux pane)
    bash "$PROJECT_ROOT/scripts/run_train_tmux.sh" start
    echo "log: $LOG_FILE"
    echo "attach: tmux attach -t $TMUX_SESSION"
    ;;
  stop|attach|status)
    bash "$PROJECT_ROOT/scripts/run_train_tmux.sh" "$1"
    ;;
  *)
    echo "usage: $0 {start|stop|attach|status}" >&2
    exit 2
    ;;
esac
