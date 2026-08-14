#!/usr/bin/env python3
"""Convert pass3-adjudicated zh texts to simplified Chinese and rebuild
phonemes/BERT for the entries whose text actually changed (zhconv).

Run: /home/mtr/miniconda3/envs/gptsovits/bin/python scripts/fix_pass3_simplify.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HQ = PROJECT_ROOT / "data" / "dataset_hq"
ANNOTATION = HQ / "annotation.list"
MANIFEST = HQ / "manifest.json"
BACKUP_DIR = HQ / "backup_simplify_20260813"

EXP_DIR = Path("/home/mtr/tt/GPT-SoVITS/logs/huayin-all")
NAME2TEXT = EXP_DIR / "2-name2text.txt"
BERT_DIR = EXP_DIR / "3-bert"
GS_ROOT = Path("/home/mtr/tt/GPT-SoVITS")
GETTEXT = GS_ROOT / "GPT_SoVITS/prepare_datasets/1-get-text.py"
PY = "/home/mtr/miniconda3/envs/gptsovits/bin/python"


def main() -> int:
    from zhconv import convert

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed: dict[str, str] = {}
    for e in manifest:
        if e.get("text_source") == "pass3":
            t = str(e.get("text") or "")
            s = convert(t, "zh-cn")
            if s != t:
                changed[e["dataset_path"].split("/")[-1]] = s
                e["text"] = s
    print(f"pass3 texts to simplify: {len(changed)}")
    if not changed:
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ANNOTATION, BACKUP_DIR / "annotation.list")
    shutil.copy2(MANIFEST, BACKUP_DIR / "manifest.json")
    shutil.copy2(NAME2TEXT, BACKUP_DIR / "2-name2text.txt")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ANNOTATION.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        name = line.split("|", 1)[0].split("/")[-1]
        if name in changed:
            prefix, _, _, _ = line.split("|", 3)
            out.append(f"{prefix}|{changed[name]}")
        else:
            out.append(line)
    ANNOTATION.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"annotation.list updated ({len(changed)} lines)")

    tmp_list = HQ / "_simplify_changed.list"
    ann_by_name = {l.split("|", 1)[0].split("/")[-1]: l for l in
                   ANNOTATION.read_text(encoding="utf-8").splitlines() if l.strip()}
    spk = ann_by_name[next(iter(changed))].split("|")[1]
    tmp_list.write_text("\n".join(
        f"audio/{n}|{spk}|ZH|{t}" for n, t in sorted(changed.items())
    ) + "\n", encoding="utf-8")

    for name in changed:
        (BERT_DIR / f"{name}.pt").unlink(missing_ok=True)

    reg_dir = EXP_DIR.parent / f"pass3_simplify_{int(time.time())}"
    (reg_dir / "3-bert").mkdir(parents=True, exist_ok=True)
    procs = []
    for i, gpu in enumerate(["0", "1"]):
        env = {
            **os.environ,
            "inp_text": str(tmp_list),
            "inp_wav_dir": str(HQ / "audio"),
            "exp_name": "huayin-all",
            "opt_dir": str(reg_dir),
            "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "is_half": "True",
            "version": "v2Pro",
            "i_part": str(i),
            "all_parts": "2",
            "_CUDA_VISIBLE_DEVICES": gpu,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
        }
        procs.append(subprocess.Popen([PY, "-s", str(GETTEXT)], cwd=str(GS_ROOT), env=env))
    codes = [p.wait() for p in procs]

    shard_map: dict[str, str] = {}
    for i in range(2):
        part = reg_dir / f"2-name2text-{i}.txt"
        if part.is_file():
            for line in part.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    shard_map[line.split("\t", 1)[0]] = line
    moved = 0
    for pt in (reg_dir / "3-bert").glob("*.pt"):
        shutil.move(str(pt), str(BERT_DIR / pt.name))
        moved += 1
    shutil.rmtree(reg_dir, ignore_errors=True)

    missing = [n for n in changed if n not in shard_map]
    if any(c != 0 for c in codes) or missing:
        raise RuntimeError(f"regeneration incomplete: codes={codes} missing={missing[:20]}")
    tmp_list.unlink(missing_ok=True)

    lines = NAME2TEXT.read_text(encoding="utf-8").splitlines()
    out = []
    replaced = 0
    for line in lines:
        if not line.strip():
            continue
        name = line.split("\t", 1)[0]
        if name in changed:
            out.append(shard_map[name])
            replaced += 1
            continue
        out.append(line)
    NAME2TEXT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"name2text merged: replaced {replaced}, moved bert {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
