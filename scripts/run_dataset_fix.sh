#!/usr/bin/env bash
# 全量数据集修复流水线（tmux / 后台可跑）：
#   1) 全量重转写 data/dataset_hq/audio（缓存按文件名复用，缺啥转啥）
#   2) LLM 校对 zh 转录，--only-missing 自动补漏，直到全部覆盖或连续无进展
# 用法: bash scripts/run_dataset_fix.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/mtr/miniconda3/envs/gptsovits/bin/python
LOG=logs/dataset_fix.log
: > "$LOG"

echo "[1/2] retranscribe all (dataset_hq/audio, gpus 0,1)" | tee -a "$LOG"
"$PY" scripts/retranscribe_dataset.py \
  --audio-dir data/dataset_hq/audio \
  --out-json data/dataset_clean/asr_retranscribed.json \
  --gpus 0,1 2>&1 | tee -a "$LOG"
echo "retranscribe exit=$?" | tee -a "$LOG"

echo "[2/2] LLM correct zh transcripts (auto retry until done)" | tee -a "$LOG"
prev=-1
for round in $(seq 1 12); do
  "$PY" scripts/correct_transcripts.py --workers 4 --batch 20 --only-missing 2>&1 | tee -a "$LOG"
  missing=$("$PY" - <<'PYEOF'
import json
from pathlib import Path
recs = json.load(open('data/dataset_clean/asr_retranscribed.json'))
zh = {Path(r['path']).name for r in recs if r['lang'] == 'zh'}
cor = {Path(k).name for k in json.load(open('data/dataset_clean/text_corrected.json'))}
rej = {Path(k).name for k in json.load(open('data/dataset_clean/correct_rejected.json'))}
print(len(zh - cor - rej))
PYEOF
  )
  echo "round $round: missing=$missing" | tee -a "$LOG"
  [ "$missing" -eq 0 ] && break
  if [ "$missing" -eq "$prev" ]; then
    echo "no progress (missing=$missing), stopping" | tee -a "$LOG"
    break
  fi
  prev=$missing
done
echo "ALL DONE" | tee -a "$LOG"
