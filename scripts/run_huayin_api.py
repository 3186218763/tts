#!/usr/bin/env python3
"""Start GPT-SoVITS api_v2.py with the trained Huayin v2Pro weights."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOVITS_ROOT = Path("/home/mtr/tt/GPT-SoVITS")
DEFAULT_GPT_MODEL = PROJECT_ROOT / "model" / "huayin-gpt.ckpt"
DEFAULT_SOVITS_MODEL = PROJECT_ROOT / "model" / "huayin-sovits.pth"

# Prefer locked recipe if present (configs/huayin_precision.yaml).
try:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from huayin_precision_recipe import delivery_paths  # type: ignore

    _delivery = delivery_paths()
    DEFAULT_GPT_MODEL = _delivery["gpt_model"]
    DEFAULT_SOVITS_MODEL = _delivery["sovits_model"]
except Exception:
    pass


def audio_compatibility_source() -> str:
    """Return a sitecustomize patch for TorchCodec/libffi-incompatible environments."""
    return '''
import numpy as _np
import soundfile as _sf
import torch as _torch
import torchaudio as _torchaudio


def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True,
                    channels_first=True, format=None, buffer_size=4096,
                    backend=None):
    frames = num_frames if num_frames >= 0 else -1
    audio, sample_rate = _sf.read(
        path,
        dtype="float32",
        always_2d=True,
        start=frame_offset,
        frames=frames,
    )
    array = audio.T if channels_first else audio
    return _torch.from_numpy(_np.ascontiguousarray(array)), int(sample_rate)


_torchaudio.load = _soundfile_load
'''


def runtime_config(
    sovits_root: Path,
    gpt_model: Path,
    sovits_model: Path,
    *,
    device: str = "cuda",
    full_precision: bool = False,
) -> dict:
    """Build the small config consumed by GPT-SoVITS TTS_Config."""
    root = sovits_root.expanduser().resolve()
    return {
        "custom": {
            "device": device,
            "is_half": device.startswith("cuda") and not full_precision,
            "version": "v2Pro",
            "t2s_weights_path": str(gpt_model.expanduser().resolve()),
            "vits_weights_path": str(sovits_model.expanduser().resolve()),
            "bert_base_path": str(
                root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
            ),
            "cnhuhbert_base_path": str(
                root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"
            ),
        }
    }


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sovits-root", type=Path, default=DEFAULT_SOVITS_ROOT)
    parser.add_argument("--gpt-model", type=Path, default=DEFAULT_GPT_MODEL)
    parser.add_argument("--sovits-model", type=Path, default=DEFAULT_SOVITS_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the GPT-SoVITS environment",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.sovits_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"GPT-SoVITS root not found: {root}")
    gpt_model = _require_file(args.gpt_model, "GPT checkpoint")
    sovits_model = _require_file(args.sovits_model, "SoVITS checkpoint")

    with tempfile.TemporaryDirectory(prefix="huayin_api_") as temp_dir:
        config_path = Path(temp_dir) / "tts_infer.yaml"
        (Path(temp_dir) / "sitecustomize.py").write_text(
            audio_compatibility_source(), encoding="utf-8"
        )
        config_path.write_text(
            yaml.safe_dump(
                runtime_config(
                    root,
                    gpt_model,
                    sovits_model,
                    device=args.device,
                    full_precision=args.full_precision,
                ),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        paths = [
            root,
            root / "GPT_SoVITS",
            root / "GPT_SoVITS" / "BigVGAN",
            root / "tools",
        ]
        env = os.environ.copy()
        env["version"] = "v2Pro"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [temp_dir] + [str(path) for path in paths]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        command = [
            args.python,
            "api_v2.py",
            "--bind_addr",
            args.host,
            "--port",
            str(args.port),
            "--tts_config",
            str(config_path),
        ]
        print(f"starting Huayin API on http://{args.host}:{args.port}", flush=True)
        print(f"GPT checkpoint: {gpt_model}", flush=True)
        print(f"SoVITS checkpoint: {sovits_model}", flush=True)
        return subprocess.run(command, cwd=root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
