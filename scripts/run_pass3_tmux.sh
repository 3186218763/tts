#!/usr/bin/env bash
# 中度组三方交叉验证流水线（tmux 后台跑）：
#   1) pass3 转写（无 prompt，温度 0.2）只处理 0.60<=sim(p1,p2)<0.85 且仍在数据集中的 zh
#   2) 三方裁决（pass1/pass2/pass3），不一致剔除、两遍一致取一致文本
#   3) 应用：更新 annotation.list / manifest / 2-name2text / 3-bert（只重建变化的文本）
#   4) 一致性校验
# 用法: bash scripts/run_pass3_tmux.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/mtr/miniconda3/envs/gptsovits/bin/python
SESSION=huayin-pass3
LOG=logs/pass3_pipeline.log

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; attaching..."
  exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n pass3 "bash -lc '
set -uo pipefail
cd /home/mtr/tt/tts
: > logs/pass3_pipeline.log
echo \"=== [1/4] pass3 ASR (mid zh, gpus 0,1) === $(date) \" | tee -a logs/pass3_pipeline.log
/home/mtr/miniconda3/envs/gptsovits/bin/python scripts/retranscribe_pass3.py --gpus 0,1 2>&1 | tee -a logs/pass3_zh.log
echo \"pass3 exit=\$?\" | tee -a logs/pass3_pipeline.log

echo \"=== [2/4] 3-way adjudication + apply === $(date)\" | tee -a logs/pass3_pipeline.log
/home/mtr/miniconda3/envs/gptsovits/bin/python scripts/adjudicate_pass3.py --apply 2>&1 | tee -a logs/pass3_adjudicate.log
echo \"adjudicate exit=\$?\" | tee -a logs/pass3_pipeline.log

echo \"=== [3/4] auto validation === $(date)\" | tee -a logs/pass3_pipeline.log
/home/mtr/miniconda3/envs/gptsovits/bin/python scripts/validate_hq_dataset.py --config configs/dataset_hq.yaml --dataset-dir data/dataset_hq 2>&1 | tee -a logs/pass3_validate.log
echo \"validate exit=\$?\" | tee -a logs/pass3_pipeline.log

echo \"=== [4/4] final consistency check === $(date)\" | tee -a logs/pass3_pipeline.log
/home/mtr/miniconda3/envs/gptsovits/bin/python - <<\"PYEOF\" 2>&1 | tee -a logs/pass3_pipeline.log
import json
from pathlib import Path
HQ=Path(\"data/dataset_hq\"); EXP=Path(\"/home/mtr/tt/GPT-SoVITS/logs/huayin-all\")
ann=[l.split(\"|\")[0].split(\"/\")[-1] for l in HQ.joinpath(\"annotation.list\").read_text().splitlines() if l.strip()]
nt={l.split(chr(9))[0]:l for l in EXP.joinpath(\"2-name2text.txt\").read_text().splitlines() if chr(9) in l}
bert={p.name[:-3] for p in EXP.joinpath(\"3-bert\").glob(\"*.pt\")}
zh=[n for n in ann if n.endswith(\"_zh.wav\")]
miss_nt=[n for n in ann if n not in nt]
miss_bert=[n for n in zh if n not in bert]
unk=[n for n,line in nt.items() if n in ann and \"UNK\" in line.split(chr(9))[1]]
m=json.loads(HQ.joinpath(\"manifest.json\").read_text())
mn={e[\"dataset_path\"].split(\"/\")[-1] for e in m}
print(f\"annotation={len(ann)} (zh={len(zh)}) name2text={len(nt)} bert={len(bert)} manifest={len(mn)}\")
print(f\"missing name2text={len(miss_nt)} missing bert={len(miss_bert)} unk={len(unk)}\")
print(\"miss_nt sample:\", miss_nt[:5]); print(\"miss_bert sample:\", miss_bert[:5])
assert not miss_nt and not miss_bert and not unk, \"consistency FAIL\"
print(\"CONSISTENCY OK\")
PYEOF
echo \"=== ALL DONE === $(date)\" | tee -a logs/pass3_pipeline.log
'"

echo "tmux session $SESSION started; attach: tmux attach -t $SESSION"
echo "log: logs/pass3_pipeline.log (stages) / logs/pass3_zh.log (ASR progress)"
