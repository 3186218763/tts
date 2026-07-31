"""花音训练数据管线——把 data/raw 的原始音频加工成 GPT-SoVITS 可用训练集。

对应 spec §3.2 步骤 ③→⑥：
  ③ separate  UVR5/BS-Roformer 人声分离（去 BGM，复用 GPT-SoVITS 已下载的权重）
  ④ slice     按 RMS 静音切分（复用 GPT-SoVITS 的 slicer2，纯 numpy）
  ⑤ asr       faster-whisper 转写 + 自动识别语种
  ⑥ dataset   汇总为 data/dataset/audio/*.wav + annotation.list（spec §3.4 格式）

每个阶段幂等：输出已存在则跳过。可用 --stage 单独运行某一阶段。

用法：
  python scripts/build_dataset.py                 # 跑完整管线
  python scripts/build_dataset.py --stage slice   # 只跑切分
  python scripts/build_dataset.py --whisper-model large-v3-turbo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 复用 GPT-SoVITS 的纯 numpy 切分器（避免在 my_utils 里引入 gradio）
SOVITS_TOOLS = Path("/home/mtr/tt/GPT-SoVITS/tools")
if str(SOVITS_TOOLS) not in sys.path:
    sys.path.insert(0, str(SOVITS_TOOLS))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAV_DIR = PROJECT_ROOT / "data" / "wav"          # ② 的产物：48kHz 单声道 WAV
VOCALS_DIR = PROJECT_ROOT / "data" / "vocals"    # ③ 的产物：分离后人声
SLICES_DIR = PROJECT_ROOT / "data" / "slices"    # ④ 的产物：5-15s 短句
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"  # ⑥ 的产物：最终训练集
ASR_RESULTS = PROJECT_ROOT / "data" / "asr_results.json"  # ⑤ 转写结果缓存

UVR5_WEIGHTS = Path("/home/mtr/tt/GPT-SoVITS/tools/uvr5/uvr5_weights")
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # 当前最强人声分离模型

# GPT-SoVITS 标注语种标签：faster-whisper ISO 码 → 训练格式
LANG_TAG = {"ja": "JP", "zh": "ZH", "yue": "ZH", "en": "EN", "ko": "KR"}


# ─────────────────────────── 工具函数 ───────────────────────────

def load_audio(file: str, sr: int) -> "numpy.ndarray":
    """ffmpeg 子进程解码音频为单声道 float32，指定采样率。

    等价于 GPT-SoVITS tools/my_utils.load_audio，但不依赖 gradio/pandas。
    """
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", file, "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).flatten()


# ─────────────────────────── ③ 人声分离 ───────────────────────────

def separate_vocals() -> None:
    import numpy as np
    from audio_separator.separator import Separator

    VOCALS_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(WAV_DIR.glob("*.wav"))
    if not wavs:
        sys.exit(f"✗ {WAV_DIR} 下没有 WAV，请先跑步骤 ②（或 bili_download + ffmpeg）")

    separator = Separator(
        output_dir=str(VOCALS_DIR),
        output_format="wav",
        sample_rate=48000,
        model_file_dir=str(UVR5_WEIGHTS),  # 复用 GPT-SoVITS 已下载的权重
        log_level=logging.WARNING,
    )
    print(f"加载人声分离模型: {VOCAL_MODEL}")
    separator.load_model(VOCAL_MODEL)

    for wav in wavs:
        # audio-separator 输出命名：{stem}_(Vocals)_{model}.wav
        if list(VOCALS_DIR.glob(f"{wav.stem}*_(Vocals)*.wav")):
            print(f"  跳过（已存在）: {wav.name}")
            continue
        print(f"  分离: {wav.name}")
        outputs = separator.separate(str(wav))
        # BS-Roformer 2-stem：输出含 "(Vocals)" 和 "(Instrumental)"
        vocal_out = next((o for o in outputs if "(Vocals)" in o), outputs[0] if outputs else None)
        if vocal_out is None:
            print(f"    ✗ 未产生人声输出")
        else:
            print(f"    ✓ {Path(vocal_out).name}")


# ─────────────────────────── ④ 静音切分 ───────────────────────────

def slice_audio() -> None:
    import numpy as np
    from scipy.io import wavfile
    from slicer2 import Slicer  # GPT-SoVITS 的纯 numpy 切分器

    SLICES_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SLICES_DIR.glob("*.wav"))
    if existing:
        print(f"  跳过（已存在 {len(existing)} 个切片）")
        return
    # audio-separator 输出命名：{stem}_(Vocals)_{model}.wav，只取人声轨
    vocals = sorted(VOCALS_DIR.glob("*_(Vocals)*.wav"))
    if not vocals:
        sys.exit(f"✗ {VOCALS_DIR} 下没有 (Vocals) 文件，请先跑 separate 阶段")

    slicer = Slicer(
        sr=32000,
        threshold=-35,      # 小于此 dB 视为静音候选切点
        min_length=3000,    # 每段最短（单位与 slicer2 内部一致，≈3s）
        min_interval=300,   # 最短切分间隔
        hop_size=10,        # 音量曲线计算步长
        max_sil_kept=500,   # 切完后最多保留多长静音
    )

    total = 0
    for vf in vocals:
        print(f"  切分: {vf.name}")
        audio = load_audio(str(vf), 32000)
        for chunk, start, end in slicer.slice(audio):
            peak = float(np.abs(chunk).max())
            if peak > 1:
                chunk = chunk / peak
            chunk = (chunk * 0.9 * 0.25) + (1 - 0.25) * chunk  # _max=0.9, alpha=0.25
            out = SLICES_DIR / f"{vf.stem}__{start:010d}_{end:010d}.wav"
            wavfile.write(str(out), 32000, (chunk * 32767).astype(np.int16))
            total += 1
    print(f"  共产出 {total} 个切片 → {SLICES_DIR}")


# ─────────────────────────── ⑤ Whisper 转写 ───────────────────────────

def transcribe(model_size: str) -> list[dict]:
    import torch
    from faster_whisper import WhisperModel

    SLICES_DIR.mkdir(parents=True, exist_ok=True)
    slices = sorted(SLICES_DIR.glob("*.wav"))
    if not slices:
        sys.exit(f"✗ {SLICES_DIR} 下没有切片，请先跑 slice 阶段")

    from faster_whisper.utils import download_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # size 名（如 large-v3）走 Systran 官方下载；含 "/" 视为 HF repo id 或本地路径
    model_path = model_size if ("/" in model_size or os.path.isdir(model_size)) else download_model(model_size)
    print(f"加载 faster-whisper [{model_size}] → {model_path} on {device} ...")
    model = WhisperModel(model_path, device=device, compute_type="float16" if device == "cuda" else "int8")

    results = []
    for i, wav in enumerate(slices, 1):
        segments, info = model.transcribe(
            str(wav), beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            language=None,  # 自动检测语种
        )
        text = "".join(s.text for s in segments).strip()
        lang = (info.language or "").lower()
        print(f"  [{i}/{len(slices)}] {wav.name}  ({lang})  {text[:50]}")
        results.append({"path": str(wav), "lang": lang, "text": text})

    ASR_RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ 转写结果已保存: {ASR_RESULTS}")
    return results


# ─────────────────────────── ⑥ 汇总训练集 ───────────────────────────

def build_dataset(transcripts: list[dict] | None = None) -> Path:
    if transcripts is None:
        if not ASR_RESULTS.exists():
            sys.exit(f"✗ 没有 ASR 结果（{ASR_RESULTS}），请先跑 --stage asr")
        transcripts = json.loads(ASR_RESULTS.read_text(encoding="utf-8"))
        print(f"  从 {ASR_RESULTS} 载入 {len(transcripts)} 条转写结果")

    audio_out = DATASET_DIR / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    lines = []
    for item in transcripts:
        text = item["text"].replace("|", "/").strip()
        src = Path(item["path"])
        tag = LANG_TAG.get(item["lang"], item["lang"] or "JP")

        # 过滤：太短或没听出字的片段
        if len(text) < 2 or src.stat().st_size < 8_000:  # <~8KB≈极短
            skipped += 1
            continue

        kept += 1
        name = f"{kept:03d}_{tag.lower()}.wav"
        dst = audio_out / name
        shutil.copy2(src, dst)
        lines.append(f"audio/{name}|花音|{tag}|{text}")

    list_path = DATASET_DIR / "annotation.list"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ 训练集就绪: {DATASET_DIR}")
    print(f"  保留 {kept} 段，丢弃 {skipped} 段（太短/无声）")
    print(f"  标注文件: {list_path}")
    return list_path


# ─────────────────────────── 主入口 ───────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="花音训练数据管线")
    p.add_argument("--stage", choices=["all", "separate", "slice", "asr", "dataset"],
                   default="all", help="单独运行某一阶段")
    p.add_argument("--whisper-model", default="large-v3",
                   help="faster-whisper 模型 size（large-v3/medium/small）或 HF repo id（含 /）")
    args = p.parse_args()

    print(f"=== 花音训练数据管线 | stage={args.stage} ===\n")

    if args.stage in ("all", "separate"):
        print("③ 人声分离 (UVR5 BS-Roformer)")
        separate_vocals()
        print()

    if args.stage in ("all", "slice"):
        print("④ 静音切分 (slicer2)")
        slice_audio()
        print()

    transcripts = []
    if args.stage in ("all", "asr"):
        print("⑤ Whisper 转写 + 语种识别")
        transcripts = transcribe(args.whisper_model)
        print()

    if args.stage in ("all", "dataset"):
        print("⑥ 汇总训练集")
        build_dataset(transcripts or None)  # transcripts 为空时自动从 asr_results.json 载入


if __name__ == "__main__":
    main()
