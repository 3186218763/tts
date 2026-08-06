#!/usr/bin/env python3
"""Headless GPT-SoVITS fine-tune for the 花音 dataset.

Runs the same pipeline as WebUI 1A/1B/1C:
  1) text features  2) hubert+wav32k (+sv for v2Pro)
  3) semantic tokens  4) SoVITS train  5) GPT train

Usage (from GPT-SoVITS repo root, with GPTSoVits env active):

  python /home/mtr/tt/tts/scripts/train_gpt_sovits.py \\
    --exp-name huayin \\
    --list-path /home/mtr/tt/tts/data/dataset/annotation.list \\
    --wav-dir /home/mtr/tt/tts/data/dataset/audio \\
    --version v2Pro \\
    --gpus 0-1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str] | str, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    if isinstance(cmd, list):
        pretty = " ".join(cmd)
    else:
        pretty = cmd
    log(f"$ {pretty}")
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    subprocess.run(cmd, check=True, env=merged, cwd=str(cwd) if cwd else None, shell=isinstance(cmd, str))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless GPT-SoVITS training for huayin")
    p.add_argument("--sovits-root", type=Path, default=Path("/home/mtr/tt/GPT-SoVITS"))
    p.add_argument("--exp-name", default="huayin")
    p.add_argument(
        "--list-path",
        type=Path,
        default=DEFAULT_DATASET_DIR / "annotation.list",
    )
    p.add_argument(
        "--wav-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR / "audio",
        help="Directory containing wav files (basename of list paths is joined here)",
    )
    p.add_argument("--version", default="v2Pro", choices=["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"])
    p.add_argument("--gpus", default="0-1", help="Train GPU ids joined by '-', e.g. 0-1 (DDP)")
    p.add_argument(
        "--format-gpus",
        default="",
        help="Format-stage worker GPUs, e.g. 0-0-0-0-0-0-1-1-1-1-1-1 (repeat id = multi-proc/GPU). "
        "Empty = auto: 6 workers per train GPU.",
    )
    p.add_argument(
        "--format-workers-per-gpu",
        type=int,
        default=6,
        help="Workers per GPU when --format-gpus is not set.",
    )
    p.add_argument("--batch-size-s2", type=int, default=12, help="SoVITS per-GPU batch size")
    p.add_argument("--batch-size-s1", type=int, default=8, help="GPT per-GPU batch size")
    p.add_argument("--epochs-s2", type=int, default=8, help="SoVITS total epochs")
    p.add_argument("--epochs-s1", type=int, default=15, help="GPT total epochs")
    p.add_argument("--save-every-s2", type=int, default=4)
    p.add_argument("--save-every-s1", type=int, default=1)
    p.add_argument("--text-low-lr-rate", type=float, default=0.4)
    p.add_argument("--skip-format", action="store_true", help="Skip 1A/1B/1C if already done")
    p.add_argument("--skip-1a", action="store_true", help="Skip only 1A text feature step")
    p.add_argument("--skip-1b", action="store_true", help="Skip only 1B hubert/sv step")
    p.add_argument("--skip-1c", action="store_true", help="Skip only 1C semantic step")
    p.add_argument("--skip-s2", action="store_true", help="Skip SoVITS training")
    p.add_argument("--skip-s1", action="store_true", help="Skip GPT training")
    p.add_argument("--is-half", default="True", choices=["True", "False"])
    return p.parse_args()


def gpu_list(gpus: str) -> list[str]:
    return [g.strip() for g in gpus.split("-") if g.strip() != ""]


def format_gpu_list(args: argparse.Namespace) -> list[str]:
    if args.format_gpus.strip():
        return gpu_list(args.format_gpus)
    # BERT/hubert workers use roughly 2 GB each on 24 GB cards.
    base = gpu_list(args.gpus) or ["0"]
    workers_per_gpu = args.format_workers_per_gpu
    if workers_per_gpu < 1:
        raise ValueError("--format-workers-per-gpu must be positive")
    return [g for g in base for _ in range(workers_per_gpu)]


def _spawn_parts(
    root: Path,
    py: str,
    script: str,
    gpus: list[str],
    base_env: dict[str, str],
    label: str,
) -> None:
    """Launch one process per entry in gpus (ids may repeat for multi-proc/GPU)."""
    procs: list[subprocess.Popen] = []
    n = len(gpus)
    worker_threads = os.environ.get("FORMAT_WORKER_THREADS", "2")
    for i, gpu in enumerate(gpus):
        env = {
            **base_env,
            "i_part": str(i),
            "all_parts": str(n),
            "_CUDA_VISIBLE_DEVICES": str(gpu),
            # Keep each worker bounded while still using the available CPU cores.
            "OMP_NUM_THREADS": worker_threads,
            "MKL_NUM_THREADS": worker_threads,
            "OPENBLAS_NUM_THREADS": worker_threads,
            "NUMEXPR_NUM_THREADS": worker_threads,
            "TOKENIZERS_PARALLELISM": "false",
        }
        log(f"{label} part {i}/{n} on GPU {gpu}")
        procs.append(
            subprocess.Popen(
                [py, "-s", script],
                cwd=str(root),
                env={**os.environ, **env},
            )
        )
    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        raise RuntimeError(f"{label} failed with codes {codes}")


def format_step_text(root: Path, py: str, args: argparse.Namespace, exp_dir: Path) -> None:
    gpus = format_gpu_list(args)
    _spawn_parts(
        root,
        py,
        "GPT_SoVITS/prepare_datasets/1-get-text.py",
        gpus,
        {
            "inp_text": str(args.list_path),
            "inp_wav_dir": str(args.wav_dir),
            "exp_name": args.exp_name,
            "opt_dir": str(exp_dir),
            "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "is_half": args.is_half,
            "version": args.version,
        },
        "1A text",
    )

    lines: list[str] = []
    for i in range(len(gpus)):
        part = exp_dir / f"2-name2text-{i}.txt"
        if not part.is_file():
            raise RuntimeError(f"missing shard output: {part}")
        content = part.read_text(encoding="utf-8").strip("\n")
        if content:
            lines.extend(content.split("\n"))
        part.unlink(missing_ok=True)
    out = exp_dir / "2-name2text.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not any(lines):
        raise RuntimeError("1A produced empty 2-name2text.txt")
    log(f"1A done -> {out} ({len(lines)} lines)")


def format_step_hubert(root: Path, py: str, args: argparse.Namespace, exp_dir: Path) -> None:
    gpus = format_gpu_list(args)
    need_sv = args.version in {"v2Pro", "v2ProPlus"}
    for script in ["2-get-hubert-wav32k.py"] + (["2-get-sv.py"] if need_sv else []):
        stage_env = {
            "inp_text": str(args.list_path),
            "inp_wav_dir": str(args.wav_dir),
            "exp_name": args.exp_name,
            "opt_dir": str(exp_dir),
            "cnhubert_base_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "sv_path": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
            "is_half": args.is_half,
        }
        if script == "2-get-sv.py":
            # TorchCodec needs the system libffi ABI on this Conda/PyTorch build.
            system_libffi = Path("/lib/x86_64-linux-gnu/libffi.so.7")
            if system_libffi.is_file():
                previous_preload = os.environ.get("LD_PRELOAD", "")
                stage_env["LD_PRELOAD"] = os.pathsep.join(
                    value for value in (str(system_libffi), previous_preload) if value
                )
        _spawn_parts(
            root,
            py,
            f"GPT_SoVITS/prepare_datasets/{script}",
            gpus,
            stage_env,
            f"1B {script}",
        )
    log("1B hubert/sv done")


def _annotation_names(list_path: Path) -> set[str]:
    names: set[str] = set()
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, *_ = line.split("|", 1)
        names.add(Path(name).name)
    return names


def _names_from_text_features(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.split("\t", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if "\t" in line
    }


def _names_from_feature_dir(path: Path, *, suffix: str = "") -> set[str]:
    if not path.is_dir():
        return set()
    if suffix:
        return {item.name[: -len(suffix)] for item in path.glob(f"*{suffix}") if item.is_file()}
    return {item.name for item in path.iterdir() if item.is_file()}


def validate_training_features(exp_dir: Path, list_path: Path, version: str) -> dict[str, int]:
    """Fail before S2 when preprocessing did not produce the required feature set."""
    expected_names = _annotation_names(list_path)
    text_names = _names_from_text_features(exp_dir / "2-name2text.txt")
    hubert_names = _names_from_feature_dir(exp_dir / "4-cnhubert", suffix=".pt")
    wav_names = _names_from_feature_dir(exp_dir / "5-wav32k")
    required = [text_names, hubert_names, wav_names]
    summary = {
        "expected": len(expected_names),
        "text": len(text_names),
        "hubert": len(hubert_names),
        "wav32k": len(wav_names),
    }

    if version in {"v2Pro", "v2ProPlus"}:
        sv_names = _names_from_feature_dir(exp_dir / "7-sv_cn", suffix=".pt")
        required.append(sv_names)
        summary["sv"] = len(sv_names)

    ready_names = expected_names.intersection(*required) if required else set()
    summary["ready"] = len(ready_names)
    if ready_names != expected_names:
        sv_detail = f"7-sv_cn={summary.get('sv', 'n/a')}/{summary['expected']}; "
        raise RuntimeError(
            "incomplete training features: "
            f"{sv_detail}expected={summary['expected']}; ready={summary['ready']}; "
            f"2-name2text={summary['text']}; "
            f"4-cnhubert={summary['hubert']}; 5-wav32k={summary['wav32k']}. "
            "Re-run preprocessing and inspect its per-item errors before training."
        )
    return summary


def format_step_semantic(root: Path, py: str, args: argparse.Namespace, exp_dir: Path) -> None:
    gpus = format_gpu_list(args)
    if args.version in {"v2Pro", "v2ProPlus"}:
        s2_cfg = f"GPT_SoVITS/configs/s2{args.version}.json"
        s2g = {
            "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        }[args.version]
    elif args.version == "v2":
        s2_cfg = "GPT_SoVITS/configs/s2.json"
        s2g = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    elif args.version == "v1":
        s2_cfg = "GPT_SoVITS/configs/s2.json"
        s2g = "GPT_SoVITS/pretrained_models/s2G488k.pth"
    elif args.version == "v4":
        s2_cfg = "GPT_SoVITS/configs/s2.json"
        s2g = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
    else:
        s2_cfg = "GPT_SoVITS/configs/s2.json"
        s2g = "GPT_SoVITS/pretrained_models/s2Gv3.pth"

    _spawn_parts(
        root,
        py,
        "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
        gpus,
        {
            "inp_text": str(args.list_path),
            "exp_name": args.exp_name,
            "opt_dir": str(exp_dir),
            "pretrained_s2G": s2g,
            "s2config_path": s2_cfg,
            "is_half": args.is_half,
        },
        "1C semantic",
    )

    rows = ["item_name\tsemantic_audio"]
    for i in range(len(gpus)):
        part = exp_dir / f"6-name2semantic-{i}.tsv"
        if not part.is_file():
            raise RuntimeError(f"missing shard output: {part}")
        content = part.read_text(encoding="utf-8").strip("\n")
        if content:
            rows.extend(content.split("\n"))
        part.unlink(missing_ok=True)
    out = exp_dir / "6-name2semantic.tsv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    log(f"1C done -> {out} ({len(rows) - 1} items)")


def train_s2(root: Path, py: str, args: argparse.Namespace, exp_dir: Path, tmp_dir: Path) -> None:
    if args.version in {"v2Pro", "v2ProPlus"}:
        config_file = root / f"GPT_SoVITS/configs/s2{args.version}.json"
        s2g = {
            "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        }[args.version]
        s2d = s2g.replace("s2G", "s2D")
        weight_dir = f"SoVITS_weights_{args.version}"
    elif args.version == "v2":
        config_file = root / "GPT_SoVITS/configs/s2.json"
        s2g = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        s2d = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
        weight_dir = "SoVITS_weights_v2"
    else:
        config_file = root / "GPT_SoVITS/configs/s2.json"
        s2g = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        s2d = "GPT_SoVITS/pretrained_models/s2D488k.pth"
        weight_dir = "SoVITS_weights"

    data = json.loads(config_file.read_text(encoding="utf-8"))
    (exp_dir / f"logs_s2_{args.version}").mkdir(parents=True, exist_ok=True)
    (root / weight_dir).mkdir(parents=True, exist_ok=True)

    data["train"]["batch_size"] = args.batch_size_s2
    data["train"]["epochs"] = args.epochs_s2
    data["train"]["text_low_lr_rate"] = args.text_low_lr_rate
    data["train"]["pretrained_s2G"] = s2g
    data["train"]["pretrained_s2D"] = s2d
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = args.save_every_s2
    data["train"]["gpu_numbers"] = args.gpus
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = 32
    data["model"]["version"] = args.version
    data["data"]["exp_dir"] = str(exp_dir)
    data["s2_ckpt_dir"] = str(exp_dir)
    data["save_weight_dir"] = weight_dir
    data["name"] = args.exp_name
    data["version"] = args.version

    tmp_cfg = tmp_dir / "tmp_s2.json"
    tmp_cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    s2_ckpt_dir = exp_dir / f"logs_s2_{args.version}"
    has_s2_resume = (s2_ckpt_dir / "G_233333333333.pth").is_file() and (
        s2_ckpt_dir / "D_233333333333.pth"
    ).is_file()
    log(
        "SoVITS resume: "
        + (f"found G/D checkpoint in {s2_ckpt_dir}" if has_s2_resume else "no checkpoint; use pretrained weights")
    )
    log(f"SoVITS train: bs={args.batch_size_s2} epochs={args.epochs_s2} gpus={args.gpus}")
    run([py, "-s", "GPT_SoVITS/s2_train.py", "--config", str(tmp_cfg)], cwd=root)
    log("SoVITS training finished")


def train_s1(root: Path, py: str, args: argparse.Namespace, exp_dir: Path, tmp_dir: Path) -> None:
    cfg_name = "s1longer.yaml" if args.version == "v1" else "s1longer-v2.yaml"
    config_file = root / "GPT_SoVITS" / "configs" / cfg_name
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    if args.version == "v1":
        pretrained_s1 = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        weight_dir = "GPT_weights"
    elif args.version == "v2":
        pretrained_s1 = (
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/"
            "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        )
        weight_dir = "GPT_weights_v2"
    else:
        # v2Pro / v2ProPlus / v3 / v4 share s1v3.ckpt
        pretrained_s1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        weight_dir = f"GPT_weights_{args.version}"

    (exp_dir / "logs_s1").mkdir(parents=True, exist_ok=True)
    (exp_dir / f"logs_s1_{args.version}").mkdir(parents=True, exist_ok=True)
    (root / weight_dir).mkdir(parents=True, exist_ok=True)

    data["train"]["batch_size"] = args.batch_size_s1
    data["train"]["epochs"] = args.epochs_s1
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = args.save_every_s1
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = False
    data["train"]["half_weights_save_dir"] = weight_dir
    data["train"]["exp_name"] = args.exp_name
    data["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
    data["output_dir"] = str(exp_dir / f"logs_s1_{args.version}")

    tmp_cfg = tmp_dir / "tmp_s1.yaml"
    tmp_cfg.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    s1_ckpt_dir = exp_dir / f"logs_s1_{args.version}" / "ckpt"
    s1_checkpoints = sorted(s1_ckpt_dir.glob("epoch=*-step=*.ckpt"))
    if s1_checkpoints:
        log(f"GPT resume: using newest checkpoint {s1_checkpoints[-1]}")
    else:
        log("GPT resume: no interruption checkpoint found; starting from pretrained s1v3.ckpt")

    gpus_csv = ",".join(gpu_list(args.gpus))
    env = {"_CUDA_VISIBLE_DEVICES": gpus_csv, "hz": "25hz"}
    log(f"GPT train: bs={args.batch_size_s1} epochs={args.epochs_s1} gpus={gpus_csv}")
    run([py, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(tmp_cfg)], env=env, cwd=root)
    log("GPT training finished")


def setup_pythonpath(root: Path) -> None:
    """Mirror webui.py users.pth so child scripts can import text/tools/GPT_SoVITS."""
    paths = [
        str(root),
        str(root / "GPT_SoVITS"),
        str(root / "GPT_SoVITS" / "BigVGAN"),
        str(root / "tools"),
        str(root / "tools" / "asr"),
        str(root / "tools" / "uvr5"),
    ]
    # Prefer env PYTHONPATH so Popen children inherit it even with -s.
    existing = os.environ.get("PYTHONPATH", "")
    merged = os.pathsep.join(paths + ([existing] if existing else []))
    os.environ["PYTHONPATH"] = merged
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Also drop a users.pth like official webui for any bare python -s launches.
    try:
        import site

        for site_root in site.getsitepackages():
            if "packages" in site_root and os.path.isdir(site_root):
                pth = Path(site_root) / "users.pth"
                pth.write_text("\n".join(paths) + "\n", encoding="utf-8")
                break
    except Exception as e:
        log(f"warn: could not write users.pth: {e}")


def main() -> int:
    args = parse_args()
    root = args.sovits_root.resolve()
    if not root.exists():
        raise SystemExit(f"GPT-SoVITS root not found: {root}")

    py = sys.executable
    exp_dir = root / "logs" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / "TEMP"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(root)
    os.environ["version"] = args.version
    setup_pythonpath(root)

    log(f"exp={args.exp_name} version={args.version} list={args.list_path}")
    log(f"wav_dir={args.wav_dir} gpus={args.gpus} python={py}")

    if not args.list_path.is_file():
        raise SystemExit(f"annotation list missing: {args.list_path}")
    if not args.wav_dir.is_dir():
        raise SystemExit(f"wav dir missing: {args.wav_dir}")

    if args.skip_format:
        log("skip format (1A/1B/1C)")
    else:
        if not args.skip_1a:
            format_step_text(root, py, args, exp_dir)
        else:
            log("skip 1A text")
        if not args.skip_1b:
            format_step_hubert(root, py, args, exp_dir)
        else:
            log("skip 1B hubert/sv")
        if not args.skip_1c:
            format_step_semantic(root, py, args, exp_dir)
        else:
            log("skip 1C semantic")

    if not args.skip_s2:
        feature_summary = validate_training_features(exp_dir, args.list_path, args.version)
        log(
            "training features ready: "
            f"{feature_summary['ready']}/{feature_summary['expected']} "
            f"(text={feature_summary['text']} hubert={feature_summary['hubert']} "
            f"wav32k={feature_summary['wav32k']}"
            + (f" sv={feature_summary['sv']}" if "sv" in feature_summary else "")
            + ")"
        )
        train_s2(root, py, args, exp_dir, tmp_dir)
    else:
        log("skip SoVITS train")

    if not args.skip_s1:
        train_s1(root, py, args, exp_dir, tmp_dir)
    else:
        log("skip GPT train")

    log("All done.")
    log(f"SoVITS weights: {root}/SoVITS_weights_{args.version}/")
    log(f"GPT weights:    {root}/GPT_weights_{args.version}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
