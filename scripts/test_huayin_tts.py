#!/usr/bin/env python3
"""Run local inference with the trained Huayin GPT-SoVITS v2Pro model."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOVITS_ROOT = Path("/home/mtr/tt/GPT-SoVITS")
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "model" / "huayin-ref.wav"
DEFAULT_REFERENCE_TEXT = "所以和朋友一起吃饭的话比较好哦"
DEFAULT_REFERENCE_LANGUAGE = "zh"
DEFAULT_GPT_MODEL = PROJECT_ROOT / "model" / "huayin-gpt.ckpt"
DEFAULT_SOVITS_MODEL = PROJECT_ROOT / "model" / "huayin-sovits.pth"
DEFAULT_SEED = 42
DEFAULT_TOP_K = 15
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REPETITION_PENALTY = 1.35
DEFAULT_SPEED = 1.0
DEFAULT_SPLIT_METHOD = "cut5"
DEFAULT_TEXT_LANGUAGE = "zh"
DEFAULT_DEVICE = "cuda"

# Locked recipe overrides (configs/huayin_precision.yaml).
try:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from huayin_precision_recipe import delivery_paths, inference_params  # type: ignore

    _d = delivery_paths()
    _i = inference_params()
    DEFAULT_GPT_MODEL = _d["gpt_model"]
    DEFAULT_SOVITS_MODEL = _d["sovits_model"]
    DEFAULT_REFERENCE_AUDIO = _d["ref_audio"]
    DEFAULT_REFERENCE_TEXT = str(_d["ref_text"])
    DEFAULT_REFERENCE_LANGUAGE = str(_d["ref_language"])
    DEFAULT_SEED = int(_i.get("seed", DEFAULT_SEED))
    DEFAULT_TOP_K = int(_i.get("top_k", DEFAULT_TOP_K))
    DEFAULT_TOP_P = float(_i.get("top_p", DEFAULT_TOP_P))
    DEFAULT_TEMPERATURE = float(_i.get("temperature", DEFAULT_TEMPERATURE))
    DEFAULT_REPETITION_PENALTY = float(_i.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
    DEFAULT_SPEED = float(_i.get("speed_factor", DEFAULT_SPEED))
    DEFAULT_SPLIT_METHOD = str(_i.get("text_split_method", DEFAULT_SPLIT_METHOD))
    DEFAULT_TEXT_LANGUAGE = str(_i.get("text_language", DEFAULT_TEXT_LANGUAGE))
    DEFAULT_DEVICE = str(_i.get("device", DEFAULT_DEVICE))
except Exception:
    pass

LANGUAGE_MAP = {"ZH": "zh", "JP": "ja", "EN": "en"}
SUPPORTED_LANGUAGES = ("auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize audio with the trained Huayin v2Pro model")
    parser.add_argument("text", nargs="?", help="Text to synthesize. Omit it to enter interactive mode.")
    parser.add_argument("--sovits-root", type=Path, default=DEFAULT_SOVITS_ROOT)
    parser.add_argument("--gpt-model", type=Path, default=DEFAULT_GPT_MODEL)
    parser.add_argument("--sovits-model", type=Path, default=DEFAULT_SOVITS_MODEL)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument(
        "--reference-text",
        default=None,
        help="Transcript of the reference audio. Defaults to the matching dataset annotation.",
    )
    parser.add_argument("--reference-language", choices=SUPPORTED_LANGUAGES, default=None)
    parser.add_argument("--text-language", choices=SUPPORTED_LANGUAGES, default=DEFAULT_TEXT_LANGUAGE)
    parser.add_argument("--output", type=Path, default=None, help="Output wav file or output directory")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, for example cuda or cpu")
    parser.add_argument("--full-precision", action="store_true", help="Disable fp16 inference on CUDA")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Use -1 for a random seed")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--split-method", default=DEFAULT_SPLIT_METHOD, help="GPT-SoVITS text split method")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading models")
    return parser.parse_args()


def reference_metadata(annotation_path: Path, audio_path: Path) -> tuple[str, str] | None:
    """Return the transcript and language for an audio file in annotation.list."""
    if not annotation_path.is_file():
        return None
    target_name = audio_path.name
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("|", 3)
        if len(fields) != 4 or Path(fields[0]).name != target_name:
            continue
        language = LANGUAGE_MAP.get(fields[2].strip().upper(), fields[2].strip().lower())
        return fields[3].strip(), language
    return None


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def output_path(value: Path | None) -> Path:
    if value is None:
        directory = PROJECT_ROOT / "outputs" / "tts"
        name = f"huayin_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        return directory / name
    value = value.expanduser()
    if value.suffix.lower() == ".wav":
        return value
    return value / f"huayin_{time.strftime('%Y%m%d_%H%M%S')}.wav"


def resolve_reference(args: argparse.Namespace) -> tuple[Path, str, str]:
    reference_audio = require_file(args.reference_audio, "reference audio")
    metadata = reference_metadata(DEFAULT_DATASET_DIR / "annotation.list", reference_audio)
    is_bundled_reference = reference_audio == DEFAULT_REFERENCE_AUDIO.resolve()
    reference_text = args.reference_text or (
        metadata[0] if metadata else (DEFAULT_REFERENCE_TEXT if is_bundled_reference else None)
    )
    reference_language = args.reference_language or (
        metadata[1]
        if metadata
        else (DEFAULT_REFERENCE_LANGUAGE if is_bundled_reference else None)
    )
    if not reference_text:
        raise ValueError("reference text is required; pass --reference-text for an audio outside data/dataset")
    if not reference_language:
        raise ValueError("reference language is required; pass --reference-language for an audio outside data/dataset")
    return reference_audio, reference_text, reference_language


def prepare_imports(sovits_root: Path) -> None:
    sovits_root = sovits_root.resolve()
    for path in (sovits_root, sovits_root / "GPT_SoVITS", sovits_root / "GPT_SoVITS" / "BigVGAN", sovits_root / "tools"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    os.environ["version"] = "v2Pro"
    os.chdir(sovits_root)


def patch_torchaudio_loader() -> None:
    """Use soundfile for WAV input because this environment's TorchCodec is ABI-incompatible."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def load_audio(path: str | os.PathLike[str]) -> tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        # GPT-SoVITS expects [channels, samples], matching torchaudio.load.
        return torch.from_numpy(np.ascontiguousarray(audio.T)), int(sample_rate)

    torchaudio.load = load_audio


def create_engine(args: argparse.Namespace) -> tuple[Any, tempfile.TemporaryDirectory[str]]:
    prepare_imports(args.sovits_root)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    patch_torchaudio_loader()

    runtime_dir = tempfile.TemporaryDirectory(prefix="huayin_tts_")
    config = TTS_Config(
        {
            "custom": {
                "device": args.device,
                "is_half": args.device.startswith("cuda") and not args.full_precision,
                "version": "v2Pro",
                "t2s_weights_path": str(args.gpt_model),
                "vits_weights_path": str(args.sovits_model),
                "bert_base_path": str(args.sovits_root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
                "cnhuhbert_base_path": str(args.sovits_root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
            }
        }
    )
    # TTS_Config writes its runtime state when model weights are loaded. Keep it out of the source tree.
    config.configs_path = str(Path(runtime_dir.name) / "tts_infer.yaml")
    return TTS(config), runtime_dir


def synthesize(engine: Any, text: str, args: argparse.Namespace, reference_audio: Path, reference_text: str, reference_language: str) -> tuple[int, Any]:
    request = {
        "text": text,
        "text_lang": args.text_language,
        "ref_audio_path": str(reference_audio),
        "prompt_text": reference_text,
        "prompt_lang": reference_language,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "text_split_method": args.split_method,
        "batch_size": 1,
        "split_bucket": True,
        "speed_factor": args.speed,
        "fragment_interval": 0.3,
        "seed": args.seed,
        "parallel_infer": True,
        "repetition_penalty": args.repetition_penalty,
        "streaming_mode": False,
    }
    result = list(engine.run(request))
    if not result:
        raise RuntimeError("GPT-SoVITS returned no audio")
    return result[-1]


def write_wav(path: Path, sample_rate: int, audio: Any) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def run_text(engine: Any, text: str, args: argparse.Namespace, reference_audio: Path, reference_text: str, reference_language: str) -> Path:
    output = output_path(args.output)
    started = time.monotonic()
    sample_rate, audio = synthesize(engine, text, args, reference_audio, reference_text, reference_language)
    write_wav(output, sample_rate, audio)
    duration = len(audio) / sample_rate
    print(f"saved: {output.resolve()}")
    print(f"audio: {duration:.2f}s at {sample_rate} Hz; inference: {time.monotonic() - started:.2f}s")
    return output


def main() -> int:
    args = parse_args()
    args.sovits_root = args.sovits_root.expanduser().resolve()
    args.gpt_model = require_file(args.gpt_model, "GPT checkpoint")
    args.sovits_model = require_file(args.sovits_model, "SoVITS checkpoint")
    reference_audio, reference_text, reference_language = resolve_reference(args)

    print(f"GPT:    {args.gpt_model}")
    print(f"SoVITS: {args.sovits_model}")
    print(f"reference: {reference_audio} ({reference_language}: {reference_text})")
    if args.dry_run:
        return 0

    engine, runtime_dir = create_engine(args)
    try:
        if args.text:
            run_text(engine, args.text.strip(), args, reference_audio, reference_text, reference_language)
            return 0

        print("Enter text to synthesize. Type quit or press Ctrl-D to exit.")
        while True:
            try:
                text = input("text> ").strip()
            except EOFError:
                print()
                break
            if text.lower() in {"quit", "exit", "q"}:
                break
            if text:
                run_text(engine, text, args, reference_audio, reference_text, reference_language)
        return 0
    finally:
        runtime_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
