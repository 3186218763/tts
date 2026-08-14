#!/usr/bin/env bash
# 生成 S1 训练所需特征（tmux 后台）：
#   1B: 2-get-hubert-wav32k.py + 2-get-sv.py   -> 4-cnhubert / 5-wav32k / 7-sv_cn
#   1C: 3-get-semantic.py                      -> 6-name2semantic.tsv
#   dev: make_s1_dev_split.py                  -> 2-name2text.dev.txt / 6-name2semantic.dev.tsv
# 不启动 S2/S1 训练。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/mtr/miniconda3/envs/gptsovits/bin/python
ROOT=/home/mtr/tt/GPT-SoVITS
EXP=$ROOT/logs/huayin-all
LOG=/home/mtr/tt/tts/logs/semantic_1bc.log
: > "$LOG"

echo "=== [1/3] 1B hubert/wav32k/sv + 1C semantic === $(date)" | tee -a "$LOG"
cd "$ROOT"
"$PY" -u /home/mtr/tt/tts/scripts/train_gpt_sovits.py \
  --sovits-root "$ROOT" \
  --exp-name huayin-all \
  --list-path /home/mtr/tt/tts/data/dataset_hq/annotation.list \
  --wav-dir /home/mtr/tt/tts/data/dataset_hq/audio \
  --version v2Pro \
  --gpus 0-1 \
  --format-workers-per-gpu 2 \
  --skip-1a --skip-s2 --skip-s1 2>&1 | tee -a "$LOG"
echo "1B/1C exit=$?" | tee -a "$LOG"

echo "=== [2/3] dev split === $(date)" | tee -a "$LOG"
"$PY" /home/mtr/tt/tts/scripts/make_s1_dev_split.py \
  --exp-dir "$EXP" --frac 0.04 --seed 1234 2>&1 | tee -a "$LOG"
echo "dev split exit=$?" | tee -a "$LOG"

echo "=== [3/3] feature consistency === $(date)" | tee -a "$LOG"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
from pathlib import Path
EXP = Path("/home/mtr/tt/GPT-SoVITS/logs/huayin-all")
ann = Path("/home/mtr/tt/tts/data/dataset_hq/annotation.list")
names = {l.split("|")[0].split("/")[-1] for l in ann.read_text().splitlines() if l.strip()}
def count(p):
    d = EXP / p
    return len([x for x in d.glob("*") if x.is_file()]) if d.is_dir() else None
n2t = {l.split("\t")[0] for l in (EXP/"2-name2text.txt").read_text().splitlines() if "\t" in l}
sem_lines = (EXP/"6-name2semantic.tsv").read_text().splitlines() if (EXP/"6-name2semantic.tsv").is_file() else []
sem = {l.split("\t")[0] for l in sem_lines if "\t" in l and not l.startswith("item_name")}
print(f"annotation={len(names)} name2text={len(n2t)} semantic={len(sem)}")
print(f"4-cnhubert={count('4-cnhubert')} 5-wav32k={count('5-wav32k')} 7-sv_cn={count('7-sv_cn')}")
miss_sem = names - sem
print(f"semantic missing: {len(miss_sem)}")
print("dev files:", [(EXP/f).name for f in ("2-name2text.dev.txt","6-name2semantic.dev.tsv") if (EXP/f).is_file()])
assert names == n2t == sem, "train set mismatch"
assert count('4-cnhubert') == len(names) and count('5-wav32k') == len(names), "hubert/wav32k incomplete"
assert (EXP/"2-name2text.dev.txt").is_file() and (EXP/"6-name2semantic.dev.tsv").is_file(), "dev files missing"
print("FEATURES READY")
PYEOF
echo "=== ALL DONE === $(date)" | tee -a "$LOG"
