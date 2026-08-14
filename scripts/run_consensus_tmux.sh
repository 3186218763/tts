#!/usr/bin/env bash
# 无人工高精度转写管线（tmux 后台）：
#   1) 异构双 ASR：Whisper large-v3 + FunASR Paraformer（增量缓存）
#   2) 共识过滤 → data/dataset_precision/（sim>=0.92 才保留）
#
# 用法:
#   bash scripts/run_consensus_tmux.sh
#   ASR_LANG=zh GPUS=0,1 bash scripts/run_consensus_tmux.sh
#   ASR_LANG=all bash scripts/run_consensus_tmux.sh
# 注意: 不要用环境变量名 LANG（会与系统 locale 冲突）
#
# 查看:
#   tmux attach -t huayin-consensus
#   tail -f logs/consensus_pipeline.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${GPTSOVITS_PY:-/home/mtr/miniconda3/envs/gptsovits/bin/python}"
SESSION="${TMUX_SESSION:-huayin-consensus}"
# ASR_LANG preferred; legacy CONSENSUS_LANG also accepted. Never read $LANG (locale).
LANG_OPT="${ASR_LANG:-${CONSENSUS_LANG:-zh}}"
GPUS="${GPUS:-0,1}"
ENGINE="${ENGINE:-both}"
LIMIT="${LIMIT:-0}"
LOG_DIR=logs
PIPELINE_LOG="$LOG_DIR/consensus_pipeline.log"
ASR_LOG="$LOG_DIR/consensus_asr_${LANG_OPT}.log"
BUILD_LOG="$LOG_DIR/consensus_build.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  echo "  attach: tmux attach -t $SESSION"
  echo "  kill:   tmux kill-session -t $SESSION"
  exit 1
fi

# kill stray non-tmux runs of the same job (e.g. previous nohup)
if pgrep -f "scripts/run_consensus_asr.py" >/dev/null 2>&1; then
  echo "stopping existing run_consensus_asr.py processes..."
  pkill -f "scripts/run_consensus_asr.py" || true
  sleep 2
fi

LIMIT_FLAGS=""
if [[ "$LIMIT" != "0" && -n "$LIMIT" ]]; then
  LIMIT_FLAGS="--limit $LIMIT"
fi

tmux new-session -d -s "$SESSION" -n consensus "bash -lc '
set -uo pipefail
cd /home/mtr/tt/tts
mkdir -p logs
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
{
  echo \"=== consensus pipeline start \$(date) ===\"
  echo \"LANG=$LANG_OPT GPUS=$GPUS ENGINE=$ENGINE LIMIT=$LIMIT\"
  echo \"PY=$PY\"
} | tee -a $PIPELINE_LOG

echo \"=== [1/2] dual ASR ($LANG_OPT) === \$(date)\" | tee -a $PIPELINE_LOG
$PY scripts/run_consensus_asr.py \\
  --lang $LANG_OPT \\
  --gpus $GPUS \\
  --engine $ENGINE \\
  $LIMIT_FLAGS \\
  2>&1 | tee -a $ASR_LOG
asr_rc=\${PIPESTATUS[0]}
echo \"dual ASR exit=\$asr_rc\" | tee -a $PIPELINE_LOG
if [[ \$asr_rc -ne 0 ]]; then
  echo \"ASR failed; skip build\" | tee -a $PIPELINE_LOG
  exit \$asr_rc
fi

echo \"=== [2/2] build high-precision dataset === \$(date)\" | tee -a $PIPELINE_LOG
$PY scripts/build_high_precision_dataset.py \\
  --lang $LANG_OPT \\
  --apply-links \\
  2>&1 | tee -a $BUILD_LOG
build_rc=\${PIPESTATUS[0]}
echo \"build exit=\$build_rc\" | tee -a $PIPELINE_LOG

$PY - <<\"PYEOF\" 2>&1 | tee -a $PIPELINE_LOG
import json
from pathlib import Path
rep = Path(\"data/dataset_precision/report.json\")
if rep.is_file():
    r = json.loads(rep.read_text(encoding=\"utf-8\"))
    print(\"summary:\", json.dumps(r.get(\"summary\"), ensure_ascii=False, indent=2))
    print(\"by_lang:\", json.dumps(r.get(\"by_lang\"), ensure_ascii=False, indent=2))
else:
    print(\"no report.json yet\")
PYEOF

echo \"=== ALL DONE === \$(date)\" | tee -a $PIPELINE_LOG
# keep session open for inspection
exec bash
'"

echo "tmux session $SESSION started"
echo "  attach: tmux attach -t $SESSION"
echo "  logs:   $PIPELINE_LOG"
echo "          $ASR_LOG"
echo "  out:    data/dataset_precision/"
echo "  note:   dual-ASR caches resume from data/dataset_hq/asr_consensus_*.json"
