#!/usr/bin/env python3
"""Final-model demo: synthesize 10 Chinese sentences with the last checkpoints."""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import test_huayin_tts as tts_lib
import batch_tts_compare as btc


def main() -> int:
    parser = argparse.ArgumentParser(description="Final-model 10-sentence demo")
    parser.add_argument("--gpt-model", type=Path, default=PROJECT_ROOT / "model" / "huayin-gpt.ckpt")
    parser.add_argument("--sovits-model", type=Path, default=PROJECT_ROOT / "model" / "huayin-sovits.pth")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tts_final_demo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpt = args.gpt_model.expanduser().resolve()
    sovits = args.sovits_model.expanduser().resolve()
    if not gpt.is_file():
        raise SystemExit(f"GPT checkpoint missing: {gpt}")
    if not sovits.is_file():
        raise SystemExit(f"SoVITS checkpoint missing: {sovits}")
    if not btc.REF_AUDIO.is_file():
        raise SystemExit(f"reference audio missing: {btc.REF_AUDIO}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "花音 TTS 最终模型演示",
                "",
                f"GPT 权重:   {gpt.name}（验证集最优 S1）",
                f"SoVITS 权重: {sovits.name}（S2 e48）",
                f"参考音频:   {btc.REF_AUDIO.name}（{btc.REF_TEXT}）",
                f"推理参数:   temperature={args.temperature}, top_k={args.top_k}, top_p=1.0, "
                f"repetition_penalty=1.35, seed={args.seed}",
                "",
                "句子列表（01.wav ~ 10.wav）:",
            ]
            + [f"  {i:02d}. {s}" for i, s in enumerate(btc.SENTENCES, 1)]
        ),
        encoding="utf-8",
    )
    if args.dry_run:
        print(f"DRY RUN: {gpt.name} + {sovits.name}, {len(btc.SENTENCES)} sentences -> {out_dir}")
        return 0

    engine_args = argparse.Namespace(
        sovits_root=btc.SOVITS_ROOT,
        device=args.device,
        full_precision=False,
        gpt_model=gpt,
        sovits_model=sovits,
        text_language="zh",
        top_k=args.top_k,
        top_p=1.0,
        temperature=args.temperature,
        repetition_penalty=1.35,
        speed=1.0,
        split_method="cut5",
        seed=args.seed,
        reference_audio=btc.REF_AUDIO,
        reference_text=btc.REF_TEXT,
        reference_language=btc.REF_LANG,
    )
    print(f"loading {gpt.name} + {sovits.name} ...", flush=True)
    engine, runtime_dir = tts_lib.create_engine(engine_args)
    started = time.monotonic()
    try:
        for sentence_index, sentence in enumerate(btc.SENTENCES, 1):
            sample_rate, audio = tts_lib.synthesize(
                engine, sentence, engine_args, btc.REF_AUDIO, btc.REF_TEXT, btc.REF_LANG
            )
            tts_lib.write_wav(out_dir / f"{sentence_index:02d}.wav", sample_rate, audio)
            print(f"  {sentence_index:02d}.wav {len(audio) / sample_rate:.2f}s", flush=True)
    finally:
        runtime_dir.cleanup()
        del engine
        gc.collect()

    zip_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"done in {time.monotonic() - started:.0f}s")
    print(f"folder: {out_dir}")
    print(f"zip:    {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
