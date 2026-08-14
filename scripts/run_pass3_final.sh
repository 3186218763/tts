#!/usr/bin/env bash
# 中度组最终应用 + 深度一致性校验（tmux 可跑）
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/mtr/miniconda3/envs/gptsovits/bin/python
LOG=logs/pass3_final.log
: > "$LOG"

echo "=== apply (final) === $(date)" | tee -a "$LOG"
"$PY" scripts/adjudicate_pass3.py --apply 2>&1 | tee -a "$LOG"
echo "exit=$?" | tee -a "$LOG"

echo "=== deep consistency === $(date)" | tee -a "$LOG"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
import re
from pathlib import Path
from zhconv import convert
HQ = Path("data/dataset_hq"); EXP = Path("/home/mtr/tt/GPT-SoVITS/logs/huayin-all")
ann = [l for l in HQ.joinpath("annotation.list").read_text().splitlines() if l.strip()]
ann_names = [l.split("|")[0].split("/")[-1] for l in ann]
nt = {l.split("\t")[0]: l for l in EXP.joinpath("2-name2text.txt").read_text().splitlines() if "\t" in l}
bert = {p.name[:-3] for p in EXP.joinpath("3-bert").glob("*.pt")}
zh = [n for n in ann_names if n.endswith("_zh.wav")]
badfmt = [l for l in ann if len(l.split("|")) < 4]
miss_nt = [n for n in ann_names if n not in nt]
miss_bert = [n for n in zh if n not in bert]
unk = [n for n, line in nt.items() if n in ann_names and "UNK" in line.split("\t")[1]]
m = json.loads(HQ.joinpath("manifest.json").read_text())
mn = {e["dataset_path"].split("/")[-1] for e in m}
trad = [e["dataset_path"] for e in m if e.get("text_source") == "pass3" and convert(e["text"], "zh-cn") != e["text"]]
P = re.compile(r"[\s，。！？、；：…,.!?;:·~～\-_/\\|()（）「」『』【】\"'“”]+")
def cmp(t):
    return P.sub("", convert(t or "", "zh-cn"))
ann_txt = {l.split("|")[0].split("/")[-1]: l.split("|", 3)[-1] for l in ann}
mis_txt = [n for n in zh if n in nt and cmp(ann_txt[n]) != cmp(nt[n].split("\t")[-1])]
print(f"annotation={len(ann)} (zh={len(zh)}) name2text={len(nt)} bert={len(bert)} manifest={len(mn)}")
print(f"badfmt={len(badfmt)} missing_nt={len(miss_nt)} missing_bert={len(miss_bert)} unk={len(unk)} pass3_trad={len(trad)} text_mismatch={len(mis_txt)}")
print("text mismatch sample:", mis_txt[:10])
assert not badfmt and not miss_nt and not miss_bert and not unk and not trad, "FAIL"
print("CONSISTENCY OK")
PYEOF
echo "=== DONE === $(date)" | tee -a "$LOG"
