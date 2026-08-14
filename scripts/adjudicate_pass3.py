#!/usr/bin/env python3
"""3-way ASR adjudication for mid-agreement zh clips (pass1 / pass2 / pass3).

For clips with 0.60 <= sim(p1,p2) < 0.85:
  - sim(p2,p3) >= 0.85 -> keep; final text = simplified(pass2)
  - sim(p1,p3) >= 0.85 -> keep; final text = pass1 (raw)
  - otherwise          -> drop (three independent decodes disagree)

Without --apply: only writes the report + drop/change lists.
With --apply: also updates annotation.list / manifest.json / 2-name2text.txt
and 3-bert/ (regenerates phonemes + BERT for changed texts only).

Run: /home/mtr/miniconda3/envs/gptsovits/bin/python scripts/adjudicate_pass3.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HQ = PROJECT_ROOT / "data" / "dataset_hq"
AUDIO_DIR = HQ / "audio"
ANNOTATION = HQ / "annotation.list"
MANIFEST = HQ / "manifest.json"
CROSSCHECK = HQ / "crosscheck_report.json"
PASS2 = HQ / "asr_pass2_zh.json"
PASS3 = HQ / "asr_pass3_zh.json"
REPORT = HQ / "pass3_report.json"
DROP_LIST = HQ / "drop_zh3.txt"
CHANGE_LIST = HQ / "change_zh3.txt"
BACKUP_DIR = HQ / "backup_mid_20260813"

EXP_DIR = Path("/home/mtr/tt/GPT-SoVITS/logs/huayin-all")
NAME2TEXT = EXP_DIR / "2-name2text.txt"
BERT_DIR = EXP_DIR / "3-bert"
GS_ROOT = Path("/home/mtr/tt/GPT-SoVITS")
GETTEXT = GS_ROOT / "GPT_SoVITS/prepare_datasets/1-get-text.py"
PY = "/home/mtr/miniconda3/envs/gptsovits/bin/python"

MID_MIN = 0.60
MID_MAX = 0.85
AGREE = 0.85

_PUNCT_RE = re.compile(r"[\s，。！？、；：…,.!?;:·~～\-_/\\|()（）「」『』【】\"'“”]+")


def compact(text: str) -> str:
    try:
        from zhconv import convert

        text = convert(text or "", "zh-cn")
    except Exception:
        pass
    return _PUNCT_RE.sub("", text or "")


def sim(a: str, b: str) -> float:
    x, y = compact(a), compact(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return SequenceMatcher(None, x, y).ratio()


def decide(entry: dict, t3: str | None) -> dict:
    t1, t2 = entry["text1"], entry["text2"]
    if t3 is None:
        return {"name": entry["name"], "sim12": entry["sim"], "sim13": None,
                "sim23": None, "decision": "no_pass3", "text_new": None}
    s13 = round(sim(t1, t3), 4)
    s23 = round(sim(t2, t3), 4)
    try:
        from zhconv import convert as _zhconv
        t2_s = _zhconv(t2 or "", "zh-cn")
    except Exception:
        t2_s = t2
    if s23 >= AGREE and compact(t2):
        return {"name": entry["name"], "sim12": entry["sim"], "sim13": s13, "sim23": s23,
                "decision": "keep", "text_new": t2_s}
    if s13 >= AGREE and compact(t1):
        return {"name": entry["name"], "sim12": entry["sim"], "sim13": s13, "sim23": s23,
                "decision": "keep", "text_new": t1}
    return {"name": entry["name"], "sim12": entry["sim"], "sim13": s13, "sim23": s23,
            "decision": "drop", "text_new": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="apply changes to dataset + training inputs")
    args = ap.parse_args()

    report = json.loads(CROSSCHECK.read_text(encoding="utf-8"))
    pass2 = {Path(r["path"]).name: r for r in json.loads(PASS2.read_text(encoding="utf-8"))}
    pass3 = {}
    if PASS3.is_file():
        pass3 = {Path(r["path"]).name: r for r in json.loads(PASS3.read_text(encoding="utf-8"))}

    current = {l.split("|", 1)[0].split("/")[-1]: l for l in
               ANNOTATION.read_text(encoding="utf-8").splitlines() if l.strip()}

    rows: list[dict] = []
    for e in report:
        name = e["name"]
        if name not in current or not (MID_MIN <= float(e.get("sim", 0.0)) < MID_MAX):
            continue
        r = decide(e, pass3.get(name, {}).get("text") if name in pass3 else None)
        old_line = current[name]
        old_text = old_line.split("|", 3)[-1].strip()
        r["text_old"] = old_text
        r["text_new"] = None if r["decision"] == "drop" else (
            r["text_new"] if r["decision"] == "keep" else old_text)
        if r["decision"] == "keep" and compact(r["text_new"]) == compact(old_text):
            r["changed"] = False
        else:
            r["changed"] = r["decision"] == "keep"
        rows.append(r)

    drops = [r for r in rows if r["decision"] == "drop"]
    changes = [r for r in rows if r["decision"] == "keep" and r["changed"]]
    no_p3 = [r for r in rows if r["decision"] == "no_pass3"]
    unchanged = [r for r in rows if r["decision"] == "keep" and not r["changed"]]

    print(f"mid group: {len(rows)} | drop: {len(drops)} | change: {len(changes)} | "
          f"unchanged: {len(unchanged)} | no_pass3: {len(no_p3)}")

    REPORT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def dump(path: Path, header: str, sel, fmt):
        lines = [header, ""]
        for r in sorted(sel, key=lambda x: x["name"]):
            lines.extend(fmt(r))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dump(DROP_LIST, f"# 三方ASR全部不一致（sim<{AGREE}），共 {len(drops)} 条，建议剔除",
         drops, lambda r: [f"{r['name']} | sim12={r['sim12']:.2f} sim13={r['sim13']} sim23={r['sim23']}",
                           f"  P1: {r['text_old']}"])
    dump(CHANGE_LIST, f"# 三方裁决后文本变更，共 {len(changes)} 条",
         changes, lambda r: [f"{r['name']}\n  OLD: {r['text_old']}\n  NEW: {r['text_new']}"])

    if not args.apply:
        print("dry-run only; add --apply to update dataset + training inputs")
        return 0

    # ---------- apply ----------
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in [(ANNOTATION, BACKUP_DIR / "annotation.list"),
                     (MANIFEST, BACKUP_DIR / "manifest.json"),
                     (NAME2TEXT, BACKUP_DIR / "2-name2text.txt")]:
        shutil.copy2(src, dst)
    if BERT_DIR.is_dir():
        (BACKUP_DIR / "3-bert_names.txt").write_text(
            "\n".join(sorted(p.name for p in BERT_DIR.glob("*.pt"))) + "\n", encoding="utf-8")
    print(f"backup -> {BACKUP_DIR}")

    drop_names = {r["name"] for r in drops}
    change_map = {r["name"]: r["text_new"] for r in changes}

    # annotation.list
    out_lines = []
    for line in ANNOTATION.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = line.split("|", 1)[0].split("/")[-1]
        if name in drop_names:
            continue
        if name in change_map:
            prefix, spk, lang, _ = line.split("|", 3)
            out_lines.append(f"{prefix}|{spk}|{lang}|{change_map[name]}")
        else:
            out_lines.append(line)
    ANNOTATION.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # manifest.json
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    kept = []
    for e in manifest:
        name = e.get("dataset_path", "").split("/")[-1]
        if name in drop_names:
            continue
        if name in change_map:
            e["text"] = change_map[name]
            e["text_source"] = "pass3"
        kept.append(e)
    MANIFEST.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # regenerate phonemes + BERT for changed texts.
    # Use a FRESH temp opt_dir so stale 2-name2text-<i>.txt shards in EXP_DIR
    # can never be reused by 1-get-text (it skips work when its shard exists).
    if changes:
        spk = current[changes[0]["name"]].split("|")[1]
        tmp_list = HQ / "_pass3_changed.list"
        tmp_list.write_text("\n".join(
            f"audio/{r['name']}|{spk}|ZH|{r['text_new']}" for r in sorted(changes, key=lambda x: x["name"])
        ) + "\n", encoding="utf-8")
        for name in change_map:
            (BERT_DIR / f"{name}.pt").unlink(missing_ok=True)

        reg_dir = EXP_DIR.parent / f"pass3_reg_{int(time.time())}"
        (reg_dir / "3-bert").mkdir(parents=True, exist_ok=True)
        shard_map: dict[str, str] = {}
        procs = []
        n_parts = 2
        gpus = ["0", "1"]
        for i, gpu in enumerate(gpus):
            env = {
                **os.environ,
                "inp_text": str(tmp_list),
                "inp_wav_dir": str(AUDIO_DIR),
                "exp_name": "huayin-all",
                "opt_dir": str(reg_dir),
                "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                "is_half": "True",
                "version": "v2Pro",
                "i_part": str(i),
                "all_parts": str(n_parts),
                "_CUDA_VISIBLE_DEVICES": gpu,
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
            }
            procs.append(subprocess.Popen(
                [PY, "-s", str(GETTEXT)], cwd=str(GS_ROOT), env=env))
        codes = [p.wait() for p in procs]
        for i in range(n_parts):
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
        failed = [n for n in change_map if n not in shard_map]
        if any(c != 0 for c in codes) or failed:
            raise RuntimeError(
                f"1-get-text regeneration incomplete: codes={codes} failed={failed[:20]}"
            )
        print(f"regenerated {len(shard_map)}/{len(change_map)} changed clips (bert {moved})")
        tmp_list.unlink(missing_ok=True)

    # merge 2-name2text.txt
    lines = NAME2TEXT.read_text(encoding="utf-8").splitlines()
    out = []
    replaced = 0
    for line in lines:
        if not line.strip():
            continue
        name = line.split("\t", 1)[0]
        if name in drop_names:
            continue
        if name in change_map:
            new = shard_map.get(name)
            if new:
                out.append(new)
                replaced += 1
                continue
            print(f"WARN: keep old line for {name} (regeneration missing)", flush=True)
        out.append(line)
    NAME2TEXT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"name2text merged: {len(out)} lines (replaced {replaced})")

    for name in drop_names:
        (BERT_DIR / f"{name}.pt").unlink(missing_ok=True)
    print(f"dropped {len(drop_names)} bert features")

    # stats.json
    zh_n = sum(1 for l in out_lines if "_zh.wav" in l)
    ja_n = sum(1 for l in out_lines if "_jp.wav" in l)
    stats = {
        "n_target": 20000,
        "n_eff": len(out_lines),
        "n_total": len(out_lines),
        "n_zh": zh_n,
        "n_ja": ja_n,
        "zh_shortfall": False,
        "zh_ratio_actual": round(zh_n / len(out_lines), 6) if out_lines else 0.0,
        "ready_for_train": True,
        "note": "pass3 三方裁决后",
    }
    (HQ / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"stats: zh={zh_n} ja={ja_n} total={len(out_lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
