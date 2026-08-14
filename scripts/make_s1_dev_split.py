#!/usr/bin/env python3
"""Split a held-out dev set for S1 (GPT) training from the training features.

Reads exp_dir/2-name2text.txt (name\tphones\tword2ph\ttext) and
exp_dir/6-name2semantic.tsv (name\tsemantic ids), samples `frac` of the
intersection with a fixed seed, and writes:
  exp_dir/2-name2text.dev.txt
  exp_dir/6-name2semantic.dev.tsv
Both dev files keep the exact column format of the training files.

Run: /home/mtr/miniconda3/envs/gptsovits/bin/python scripts/make_s1_dev_split.py \
       --exp-dir /home/mtr/tt/GPT-SoVITS/logs/huayin-all --frac 0.04 --seed 1234
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.04, help="dev fraction of the dataset")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    phoneme = args.exp_dir / "2-name2text.txt"
    semantic = args.exp_dir / "6-name2semantic.tsv"
    for f in (phoneme, semantic):
        if not f.is_file():
            raise SystemExit(f"missing training feature file: {f} (run 1C semantic first)")

    p_lines = [l for l in phoneme.read_text(encoding="utf-8").splitlines() if l.strip()]
    s_lines = [l for l in semantic.read_text(encoding="utf-8").splitlines() if l.strip()]
    s_header = None
    if s_lines and s_lines[0].split("\t", 1)[0] == "item_name":
        s_header = s_lines[0]
        s_lines = s_lines[1:]
    p_names = {l.split("\t", 1)[0] for l in p_lines}
    s_names = {l.split("\t", 1)[0] for l in s_lines}
    common = sorted(p_names & s_names)
    if not common:
        raise SystemExit("no overlap between 2-name2text.txt and 6-name2semantic.tsv")

    rng = random.Random(args.seed)
    dev_names = set(rng.sample(common, max(1, int(len(common) * args.frac))))
    # Shuffle so max_eval_sample "first N" is not filename-order biased.
    dev_p = [l for l in p_lines if l.split("\t", 1)[0] in dev_names]
    dev_s = [l for l in s_lines if l.split("\t", 1)[0] in dev_names]
    train_p = [l for l in p_lines if l.split("\t", 1)[0] not in dev_names]
    train_s = [l for l in s_lines if l.split("\t", 1)[0] not in dev_names]
    rng.shuffle(dev_p)
    rng.shuffle(dev_s)
    rng.shuffle(train_p)
    rng.shuffle(train_s)

    out_p = args.exp_dir / "2-name2text.dev.txt"
    out_s = args.exp_dir / "6-name2semantic.dev.tsv"
    out_train_p = args.exp_dir / "2-name2text.train.txt"
    out_train_s = args.exp_dir / "6-name2semantic.train.tsv"
    out_p.write_text("\n".join(dev_p) + "\n", encoding="utf-8")
    out_s.write_text(("\n".join([s_header] + dev_s) if s_header else "\n".join(dev_s)) + "\n", encoding="utf-8")
    out_train_p.write_text("\n".join(train_p) + "\n", encoding="utf-8")
    out_train_s.write_text(("\n".join([s_header] + train_s) if s_header else "\n".join(train_s)) + "\n", encoding="utf-8")

    print(f"train: {len(common) - len(dev_names)} | dev: {len(dev_names)} "
          f"({args.frac:.1%}, seed={args.seed})")
    print(f"dev phoneme -> {out_p}")
    print(f"dev semantic -> {out_s}")
    print(f"train phoneme (dev excluded) -> {out_train_p}")
    print(f"train semantic (dev excluded) -> {out_train_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
