#!/usr/bin/env bash
# No-human high-precision transcript pipeline.
# 1) Dual ASR (Whisper + FunASR) on dataset_hq
# 2) Consensus filter -> data/dataset_precision/
#
# Usage:
#   bash scripts/run_high_precision_pipeline.sh                 # full zh+ja
#   ASR_LANG=zh LIMIT=100 bash scripts/run_high_precision_pipeline.sh  # pilot
# 注意: 用 ASR_LANG，不要用 LANG（locale）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${GPTSOVITS_PY:-/home/mtr/miniconda3/envs/gptsovits/bin/python}"
GPUS="${GPUS:-0,1}"
ASR_LANG="${ASR_LANG:-all}"
LIMIT="${LIMIT:-0}"
ENGINE="${ENGINE:-both}"

cd "$ROOT"
echo "[1/2] dual ASR lang=$ASR_LANG limit=$LIMIT gpus=$GPUS"
extra=()
if [[ "$LIMIT" != "0" ]]; then
  extra+=(--limit "$LIMIT")
fi
"$PY" scripts/run_consensus_asr.py --lang "$ASR_LANG" --gpus "$GPUS" --engine "$ENGINE" "${extra[@]}"

echo "[2/2] build high-precision dataset"
"$PY" scripts/build_high_precision_dataset.py --lang "$ASR_LANG" --apply-links

echo "done -> data/dataset_precision/"
test -f data/dataset_precision/report.json && "$PY" -c "
import json
r=json.load(open('data/dataset_precision/report.json'))
print(json.dumps(r['summary'], ensure_ascii=False, indent=2))
print('by_lang', json.dumps(r['by_lang'], ensure_ascii=False, indent=2))
"
