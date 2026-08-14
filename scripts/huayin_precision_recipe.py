#!/usr/bin/env python3
"""Load configs/huayin_precision.yaml — the locked Huayin precision recipe.

Usage:
  python scripts/huayin_precision_recipe.py              # pretty JSON
  python scripts/huayin_precision_recipe.py --export-env # shell exports for train
  python scripts/huayin_precision_recipe.py --paths      # delivery paths only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = PROJECT_ROOT / "configs" / "huayin_precision.yaml"


def load_recipe(path: Path | None = None) -> dict[str, Any]:
    recipe_path = (path or DEFAULT_RECIPE).expanduser().resolve()
    if not recipe_path.is_file():
        raise FileNotFoundError(f"recipe not found: {recipe_path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML required: pip install pyyaml") from exc
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid recipe (expected mapping): {recipe_path}")
    return raw


def resolve_path(rel_or_abs: str | Path, root: Path | None = None) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (root or PROJECT_ROOT) / p


def delivery_paths(recipe: dict[str, Any] | None = None) -> dict[str, Path]:
    r = recipe or load_recipe()
    d = r["delivery"]
    return {
        "gpt_model": resolve_path(d["gpt_model"]),
        "sovits_model": resolve_path(d["sovits_model"]),
        "ref_audio": resolve_path(d["ref_audio"]),
        "ref_text": d["ref_text"],
        "ref_language": d["ref_language"],
    }


def inference_params(recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    r = recipe or load_recipe()
    return dict(r["inference"])


def export_train_env(recipe: dict[str, Any] | None = None) -> str:
    """Shell `export KEY=value` lines for run_precision_train_tmux.sh."""
    r = recipe or load_recipe()
    p, t = r["paths"], r["train"]
    s1, s2 = t["s1"], t["s2"]
    root = PROJECT_ROOT
    pairs = {
        "SOVITS_ROOT": p["sovits_root"],
        "TMUX_SESSION": p["tmux_session"],
        "DATASET_DIR": str(resolve_path(p["dataset_dir"], root)),
        "EXP_NAME": p["exp_name"],
        "LOG_FILE": str(resolve_path(p["log_file"], root)),
        "BATCH_SIZE_S2": s2["batch_size"],
        "BATCH_SIZE_S1": s1["batch_size"],
        "EPOCHS_S2": s2["epochs"],
        "EPOCHS_S1": s1["epochs"],
        "SAVE_EVERY_S2": s2["save_every_epoch"],
        "SAVE_EVERY_S1": s1["save_every_n_epoch"],
        "FORMAT_WORKERS_PER_GPU": t["format_workers_per_gpu"],
        "FORMAT_WORKER_THREADS": t["format_worker_threads"],
        "LR_S1": s1["lr"],
        "S1_WARMUP_STEPS": s1["warmup_steps"],
        "LR_DECAY_STEPS": s1["decay_steps"],
        "S1_DEV_FRAC": s1["dev_frac"],
        "S1_VAL_BATCHES": s1["val_batches"],
        "S1_DROPOUT": s1["dropout"],
        "S2_DROPOUT": s2["p_dropout"],
        "S1_EARLY_STOP_PATIENCE": s1["early_stop_patience"],
        "S1_EARLY_STOP_MIN_DELTA": s1["early_stop_min_delta"],
        "SKIP_FORMAT": "1" if t.get("skip_format_default") else "0",
    }
    lines = []
    for key, value in pairs.items():
        # quote for shell safety
        lines.append(f"export {key}={json.dumps(str(value))}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--export-env", action="store_true")
    ap.add_argument("--paths", action="store_true")
    ap.add_argument("--inference", action="store_true")
    args = ap.parse_args()
    recipe = load_recipe(args.recipe)
    if args.export_env:
        sys.stdout.write(export_train_env(recipe))
        return 0
    if args.paths:
        d = delivery_paths(recipe)
        out = {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.inference:
        json.dump(inference_params(recipe), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    json.dump(recipe, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
