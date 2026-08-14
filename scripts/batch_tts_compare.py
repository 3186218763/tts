#!/usr/bin/env python3
"""Batch TTS comparison: synthesize 10 Chinese sentences across S1/S2 checkpoints."""

from __future__ import annotations

import argparse
import gc
import re
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import test_huayin_tts as tts_lib

SOVITS_ROOT = Path("/home/mtr/tt/GPT-SoVITS")
MODEL_DIR = PROJECT_ROOT / "model"
REF_AUDIO = MODEL_DIR / "huayin-ref.wav"
REF_TEXT = "所以和朋友一起吃饭的话比较好哦"
REF_LANG = "zh"
DEFAULT_S1 = MODEL_DIR / "huayin-gpt.ckpt"
DEFAULT_S2 = MODEL_DIR / "huayin-sovits.pth"

SENTENCES = [
    "今天天气真不错，我们一起去公园散步吧。",
    "人工智能正在改变我们生活的方方面面。",
    "请把这份文件在明天早上八点之前交给我。",
    "窗外下着小雨，泡一杯热茶最舒服了。",
    "保持专注和耐心，很多难题都会慢慢解决。",
    "这家餐厅的招牌菜味道很不错，值得推荐。",
    "学习一门新语言，需要坚持每天练习。",
    "春天来了，花园里的花都开了。",
    "我们下个月去海边度假，记得提前订好酒店。",
    "无论遇到什么困难，都要保持积极的心态。",
]


def build_combos(s1_epochs: list[int], s2_model: str) -> list[tuple[str, Path, Path]]:
    """Legacy multi-epoch sweep under GPT-SoVITS weight dirs; prefer DEFAULT_S1/S2."""
    gpt_dir = SOVITS_ROOT / "GPT_weights_v2Pro"
    sovits_dir = SOVITS_ROOT / "SoVITS_weights_v2Pro"
    s2_tag = re.sub(r"_s\d+$", "", s2_model.replace("huayin-hq_", "").replace(".pth", ""))
    return [
        (
            f"S1-e{epoch:02d}__S2-{s2_tag}",
            gpt_dir / f"huayin-hq-e{epoch}.ckpt",
            sovits_dir / s2_model,
        )
        for epoch in s1_epochs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch TTS comparison")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tts_compare_20260812")
    parser.add_argument(
        "--s1-epochs",
        default="5,10,15,20,25,30",
        help="Comma-separated S1 epochs to compare",
    )
    parser.add_argument("--s2-model", default="huayin-sovits.pth", help="SoVITS model file name")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate files and plan only")
    args = parser.parse_args()

    combos = build_combos([int(e) for e in args.s1_epochs.split(",") if e.strip()], args.s2_model)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, gpt, sovits in combos:
        if not gpt.is_file():
            raise SystemExit(f"GPT checkpoint missing: {gpt}")
        if not sovits.is_file():
            raise SystemExit(f"SoVITS checkpoint missing: {sovits}")
    if not REF_AUDIO.is_file():
        raise SystemExit(f"reference audio missing: {REF_AUDIO}")

    readme = out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "花音 TTS 训练结果对比（2026-08-12 新数据集 huayin-hq）",
                "",
                f"参考音频: {REF_AUDIO.name}（{REF_TEXT}）",
                f"推理参数固定: top_k={args.top_k}, top_p=1.0, temperature={args.temperature}, "
                "repetition_penalty=1.35, speed=1.0",
                f"seed={args.seed}（各组合相同，保证同句可比）",
                "",
                "命名规则: S1-e05__S2-e12 表示 GPT 权重 model/huayin-gpt.ckpt",
                "          + SoVITS 权重 model/huayin-sovits.pth",
                "",
                "组合列表:",
            ]
            + [f"  {name}: gpt={gpt.name}, sovits={sovits.name}" for name, gpt, sovits in combos]
            + [
                "",
                "句子列表（每个组合文件夹内 01.wav ~ 10.wav 与下面对应）:",
            ]
            + [f"  {i:02d}. {s}" for i, s in enumerate(SENTENCES, 1)]
            + [
                "",
                "听法建议: 先横向听同一条句子在 8 个文件夹里的差别，挑出整体最稳的组合；",
                "         S1 决定音色与韵律，S2 决定音质细节。",
            ]
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"DRY RUN: {len(combos)} combos x {len(SENTENCES)} sentences = {len(combos) * len(SENTENCES)} clips")
        for name, gpt, sovits in combos:
            print(f"  {name}: {gpt.name} + {sovits.name}")
        print(f"out dir: {out_dir}")
        return 0

    engine_args = argparse.Namespace(
        sovits_root=SOVITS_ROOT,
        device=args.device,
        full_precision=False,
        gpt_model=None,
        sovits_model=None,
        text_language="zh",
        top_k=args.top_k,
        top_p=1.0,
        temperature=args.temperature,
        repetition_penalty=1.35,
        speed=1.0,
        split_method="cut5",
        seed=args.seed,
        reference_audio=REF_AUDIO,
        reference_text=REF_TEXT,
        reference_language=REF_LANG,
    )
    total_started = time.monotonic()
    for combo_index, (name, gpt, sovits) in enumerate(combos, 1):
        engine_args.gpt_model = gpt
        engine_args.sovits_model = sovits
        combo_dir = out_dir / name
        combo_dir.mkdir(exist_ok=True)
        print(f"[{combo_index}/{len(combos)}] loading {name} ...", flush=True)
        engine, runtime_dir = tts_lib.create_engine(engine_args)
        try:
            for sentence_index, sentence in enumerate(SENTENCES, 1):
                started = time.monotonic()
                sample_rate, audio = tts_lib.synthesize(
                    engine, sentence, engine_args, REF_AUDIO, REF_TEXT, REF_LANG
                )
                out_wav = combo_dir / f"{sentence_index:02d}.wav"
                tts_lib.write_wav(out_wav, sample_rate, audio)
                duration = len(audio) / sample_rate
                print(
                    f"  {sentence_index:02d}.wav {duration:.2f}s in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
        finally:
            runtime_dir.cleanup()
        del engine
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    zip_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"done in {time.monotonic() - total_started:.0f}s")
    print(f"folder: {out_dir}")
    print(f"zip:    {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
