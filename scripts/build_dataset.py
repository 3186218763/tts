"""花音训练数据管线——把原始音频加工成 GPT-SoVITS 可用训练集。

管线阶段：
  ③ separate  按白名单进行 UVR5/BS-Roformer 人声分离
  ④ slice     按 RMS 静音逐文件增量切分
  ⑤ filter    用音高、能量和时长特征标注不合格切片
  ⑥ asr       仅增量转写 filter 保留的切片
  ⑦ dataset   严格过滤语言和幻觉文本，汇总训练集

各阶段均支持断点续跑。可用 --stage 单独运行某一阶段。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 复用 GPT-SoVITS 的纯 numpy 切分器（避免在 my_utils 里引入 gradio）
SOVITS_TOOLS = Path("/home/mtr/tt/GPT-SoVITS/tools")
if str(SOVITS_TOOLS) not in sys.path:
    sys.path.insert(0, str(SOVITS_TOOLS))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAV_DIR = PROJECT_ROOT / "data" / "wav"
VOCALS_DIR = PROJECT_ROOT / "data" / "vocals"
SLICES_DIR = PROJECT_ROOT / "data" / "slices"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
TRAINING_ASSETS = PROJECT_ROOT / "data" / "training_assets.txt"
FILTER_RESULTS = PROJECT_ROOT / "data" / "filter_results.json"
SPEAKER_RESULTS = PROJECT_ROOT / "data" / "speaker_results.json"
ASR_RESULTS = PROJECT_ROOT / "data" / "asr_results.json"
ALIGNMENT_RESULTS = PROJECT_ROOT / "data" / "alignment_results.json"

# Local text-quality helpers (same package directory when run as script).
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_text_quality as text_quality  # noqa: E402

UVR5_WEIGHTS = Path("/home/mtr/tt/GPT-SoVITS/tools/uvr5/uvr5_weights")
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
SEPARATION_CHUNK_SECONDS = 30 * 60

# 只允许花音实际会在素材中使用的语言。yue/ko 等视为 Whisper 幻觉信号。
LANG_TAG = {"ja": "JP", "zh": "ZH", "en": "EN"}
GENERATED_AUDIO_RE = re.compile(r"^\d+_[a-z0-9_-]+\.wav$", re.IGNORECASE)
ASR_QUALITY_VERSION = 1
MIN_LANGUAGE_PROBABILITY = 0.50
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROBABILITY = 0.60
MAX_COMPRESSION_RATIO = 2.40


@dataclass(frozen=True)
class FilterThresholds:
    """filter 阶段的可调阈值。比较符严格遵循设计文档。"""

    voiced_ratio_threshold: float = 0.75
    f0_cv_threshold: float = 0.15
    longest_voiced_threshold: float = 3.0
    min_rms_db: float = -40.0
    min_duration: float = 1.5
    max_duration: float = 30.0
    fmin: float = 80.0
    fmax: float = 400.0
    hop_length: int = 512


@dataclass(frozen=True)
class AudioFeatures:
    duration: float
    rms_db: float
    voiced_ratio: float
    f0_cv: float | None
    longest_voiced: float


# ─────────────────────────── 通用工具 ───────────────────────────


def canonical_path(path: str | Path, root: Path | None = None) -> Path:
    """把 JSON 中的相对/绝对路径统一成可比较的绝对路径。"""

    value = Path(path)
    base = PROJECT_ROOT if root is None else Path(root)
    if not value.is_absolute():
        value = base / value
    return value.resolve()


def _read_json_list(path: Path, *, required: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise SystemExit(f"✗ 缺少 {path}，请先运行前置阶段")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 必须是 JSON 数组")
    return data


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return dict(vars(value))


def _has_output_with_prefix(
    directory: Path, prefix: str, *, marker: str | None = None
) -> bool:
    """按字面前缀检查输出，避免把标题中的 [] 当成 glob 语法。"""

    if not directory.exists():
        return False
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        if path.name.startswith(prefix) and (marker is None or marker in path.name):
            return True
    return False


def _slice_end(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _has_complete_slice_output(vocal: Path) -> bool:
    """Treat a vocal as sliced only when its slice timeline reaches the end."""

    slices = [
        path
        for path in SLICES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
        and path.name.startswith(f"{vocal.stem}__")
    ] if SLICES_DIR.exists() else []
    if not slices:
        return False
    duration = _wav_duration(vocal)
    if duration <= 0:
        return True
    return max(_slice_end(path) for path in slices) >= duration * 32_000 * 0.98


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return 0.0


def _has_complete_vocal_output(source: Path) -> bool:
    """Require a near-full-duration vocal, not merely a partial output file."""

    source_duration = _wav_duration(source)
    prefix = f"{source.stem}_"
    outputs = []
    if VOCALS_DIR.exists():
        outputs = [
            path
            for path in VOCALS_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".wav"
            and path.name.startswith(prefix)
            and "_(Vocals)" in path.name
        ]
    if source_duration <= 0:
        return bool(outputs)
    return any(
        _wav_duration(output) >= source_duration * 0.98 for output in outputs
    )


def load_training_assets(
    assets_path: Path | None = None, wav_dir: Path | None = None
) -> list[Path]:
    """读取训练素材白名单，并验证每个文件都精确存在。"""

    assets_path = TRAINING_ASSETS if assets_path is None else Path(assets_path)
    wav_dir = WAV_DIR if wav_dir is None else Path(wav_dir)
    if not assets_path.exists():
        raise FileNotFoundError(f"训练素材白名单不存在：{assets_path}")

    names: list[str] = []
    seen: set[str] = set()
    for raw_line in assets_path.read_text(encoding="utf-8").splitlines():
        name = raw_line.strip()
        if not name or name.startswith("#"):
            continue
        if Path(name).name != name:
            raise ValueError(f"白名单必须只包含 WAV 文件名，不能包含路径：{name}")
        if name not in seen:
            names.append(name)
            seen.add(name)

    if not names:
        raise ValueError(f"训练素材白名单为空：{assets_path}")

    missing = [name for name in names if not (wav_dir / name).is_file()]
    if missing:
        preview = "、".join(missing[:5])
        suffix = "……" if len(missing) > 5 else ""
        raise FileNotFoundError(f"白名单中有 {len(missing)} 个 WAV 不存在：{preview}{suffix}")
    return [wav_dir / name for name in names]


def _training_asset_stems() -> set[str] | None:
    """返回白名单 stem；没有白名单时兼容测试/旧调用并不做来源限制。"""

    if not TRAINING_ASSETS.exists():
        return None
    return {path.stem for path in load_training_assets()}


def _is_training_derivative(path: Path, allowed_stems: set[str] | None) -> bool:
    if allowed_stems is None:
        return True
    source_stem, marker, _ = path.name.partition("_(Vocals)")
    return bool(marker) and source_stem in allowed_stems


def _select_training_derivatives(paths: list[Path]) -> list[Path]:
    allowed_stems = _training_asset_stems()
    return [path for path in paths if _is_training_derivative(path, allowed_stems)]


def load_audio(file: str, sr: int) -> np.ndarray:
    """通过 ffmpeg 解码为指定采样率的单声道 float32。"""

    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            file,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).flatten()


# ─────────────────────────── ③ 人声分离 ───────────────────────────


def separate_vocals(
    *, chunk_duration_seconds: int = SEPARATION_CHUNK_SECONDS
) -> None:
    VOCALS_DIR.mkdir(parents=True, exist_ok=True)
    wavs = load_training_assets()
    pending = [
        wav
        for wav in wavs
        if not _has_complete_vocal_output(wav)
    ]

    for wav in wavs:
        if wav not in pending:
            print(f"  跳过（已存在）: {wav.name}")
    if not pending:
        print(f"  白名单中的 {len(wavs)} 个文件均已分离")
        return

    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=str(VOCALS_DIR),
        output_format="wav",
        sample_rate=48000,
        model_file_dir=str(UVR5_WEIGHTS),
        output_single_stem="Vocals",
        chunk_duration=chunk_duration_seconds,
        log_level=logging.WARNING,
    )
    print(f"加载人声分离模型: {VOCAL_MODEL}")
    separator.load_model(VOCAL_MODEL)

    for wav in pending:
        print(f"  分离: {wav.name}")
        outputs = separator.separate(str(wav))
        vocal_out = next(
            (output for output in outputs if "(Vocals)" in output),
            outputs[0] if outputs else None,
        )
        if vocal_out is None:
            print("    ✗ 未产生人声输出")
        else:
            print(f"    ✓ {Path(vocal_out).name}")


# ─────────────────────────── ④ 静音切分 ───────────────────────────


def _slice_one_vocal(vocal: Path) -> int:
    from scipy.io import wavfile
    from slicer2 import Slicer

    slicer = Slicer(
        sr=32000,
        threshold=-35,
        min_length=3000,
        min_interval=300,
        hop_size=10,
        max_sil_kept=500,
    )
    audio = load_audio(str(vocal), 32000)
    total = 0
    for chunk, start, end in slicer.slice(audio):
        chunk = np.asarray(chunk)
        if chunk.size == 0:
            continue
        peak = float(np.abs(chunk).max())
        if peak > 1:
            chunk = chunk / peak
        chunk = (chunk * 0.9 * 0.25) + (1 - 0.25) * chunk
        out = SLICES_DIR / f"{vocal.stem}__{start:010d}_{end:010d}.wav"
        wavfile.write(str(out), 32000, (chunk * 32767).astype(np.int16))
        total += 1
    return total


def slice_audio(*, workers: int = 1) -> None:
    SLICES_DIR.mkdir(parents=True, exist_ok=True)
    all_vocals = sorted(VOCALS_DIR.glob("*_(Vocals)*.wav"))
    if not all_vocals:
        raise SystemExit(f"✗ {VOCALS_DIR} 下没有 (Vocals) 文件，请先跑 separate 阶段")
    vocals = _select_training_derivatives(all_vocals)
    excluded = len(all_vocals) - len(vocals)
    if excluded:
        print(f"  跳过 {excluded} 个非白名单来源的 vocal")
    if not vocals:
        raise SystemExit("✗ 当前没有白名单素材对应的 vocal，请先跑 separate 阶段")

    pending: list[Path] = []
    for vocal in vocals:
        if _has_complete_slice_output(vocal):
            print(f"  跳过（已有切片）: {vocal.name}")
        else:
            pending.append(vocal)
    if not pending:
        print(f"  {len(vocals)} 个 vocal 均已有切片")
        return

    total = 0
    worker_count = max(1, min(int(workers), len(pending)))
    if worker_count == 1:
        for vocal in pending:
            print(f"  切分: {vocal.name}")
            total += _slice_one_vocal(vocal)
    else:
        print(f"  使用 {worker_count} 个 CPU 进程并行切分")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_slice_one_vocal, vocal): vocal for vocal in pending
            }
            for future in as_completed(futures):
                vocal = futures[future]
                count = future.result()
                total += count
                print(f"  ✓ {vocal.name}: {count} 个切片")
    print(f"  本次新增 {total} 个切片 → {SLICES_DIR}")


# ─────────────────────────── ⑤ 音频过滤 ───────────────────────────


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_end = 0
    run_start: int | None = None
    for index, flag in enumerate(mask):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            if index - run_start > best_end - best_start:
                best_start, best_end = run_start, index
            run_start = None
    if run_start is not None and len(mask) - run_start > best_end - best_start:
        best_start, best_end = run_start, len(mask)
    return best_start, best_end


def extract_audio_features(
    audio: np.ndarray,
    sr: int,
    *,
    pyin_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    hop_length: int = 512,
    fmin: float = 80.0,
    fmax: float = 400.0,
) -> AudioFeatures:
    """提取歌声检测所需特征；无有效 F0 时使用 None，而非 NaN。"""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = float(samples.size / sr) if sr > 0 else 0.0
    if samples.size:
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    else:
        rms = 0.0
    rms_db = float(20.0 * math.log10(max(rms, 1e-12)))

    if samples.size == 0:
        return AudioFeatures(duration, rms_db, 0.0, None, 0.0)

    if pyin_fn is None:
        import librosa

        pyin_fn = librosa.pyin

    f0, voiced_flag, _ = pyin_fn(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )
    f0_values = np.asarray(f0, dtype=float).reshape(-1)
    voiced = np.asarray(voiced_flag, dtype=bool).reshape(-1)
    frame_count = min(f0_values.size, voiced.size)
    f0_values = f0_values[:frame_count]
    voiced = voiced[:frame_count] & np.isfinite(f0_values)

    if frame_count == 0:
        return AudioFeatures(duration, rms_db, 0.0, None, 0.0)

    voiced_ratio = float(np.count_nonzero(voiced) / frame_count)
    start, end = _longest_true_run(voiced)
    longest_voiced = float((end - start) * hop_length / sr)
    f0_cv: float | None = None
    if end > start:
        segment = f0_values[start:end]
        mean_f0 = float(np.mean(segment))
        if mean_f0 != 0:
            f0_cv = float(np.std(segment) / abs(mean_f0))

    return AudioFeatures(duration, rms_db, voiced_ratio, f0_cv, longest_voiced)


def _feature_value(features: AudioFeatures | Mapping[str, Any], name: str) -> Any:
    if isinstance(features, Mapping):
        return features[name]
    return getattr(features, name)


def classify_audio_features(
    features: AudioFeatures | Mapping[str, Any],
    thresholds: FilterThresholds,
) -> tuple[bool, str | None]:
    """返回 (keep, reason)，原因优先级为时长、能量、歌声。"""

    duration = float(_feature_value(features, "duration"))
    rms_db = float(_feature_value(features, "rms_db"))
    voiced_ratio = float(_feature_value(features, "voiced_ratio"))
    f0_cv = _feature_value(features, "f0_cv")
    longest_voiced = float(_feature_value(features, "longest_voiced"))

    if duration < thresholds.min_duration or duration > thresholds.max_duration:
        return False, "duration"
    if rms_db < thresholds.min_rms_db:
        return False, "low_energy"
    stable_sustained = (
        voiced_ratio > thresholds.voiced_ratio_threshold
        and f0_cv is not None
        and float(f0_cv) < thresholds.f0_cv_threshold
    )
    if stable_sustained or longest_voiced > thresholds.longest_voiced_threshold:
        return False, "singing"
    return True, None


def analyze_slice(path: Path, thresholds: FilterThresholds) -> dict[str, Any]:
    import librosa

    audio, sr = librosa.load(str(path), sr=None, mono=True)
    features = extract_audio_features(
        audio,
        int(sr),
        pyin_fn=librosa.pyin,
        hop_length=thresholds.hop_length,
        fmin=thresholds.fmin,
        fmax=thresholds.fmax,
    )
    keep, reason = classify_audio_features(features, thresholds)
    return {"keep": keep, "reason": reason, **asdict(features)}


def _analyze_slice_job(
    path: Path, thresholds: FilterThresholds
) -> tuple[Path, dict[str, Any]]:
    return path, analyze_slice(path, thresholds)


def filter_slices(
    thresholds: FilterThresholds | None = None,
    *,
    force: bool = False,
    analyzer: Callable[..., Any] = analyze_slice,
    workers: int = 1,
) -> list[dict[str, Any]]:
    thresholds = FilterThresholds() if thresholds is None else thresholds
    all_slices = sorted(SLICES_DIR.glob("*.wav"))
    if not all_slices:
        raise SystemExit(f"✗ {SLICES_DIR} 下没有切片，请先跑 slice 阶段")
    slices = _select_training_derivatives(all_slices)
    excluded = len(all_slices) - len(slices)
    if excluded:
        print(f"  跳过 {excluded} 个非白名单来源的切片")
    if not slices:
        raise SystemExit("✗ 当前没有白名单素材对应的切片，请先跑 slice 阶段")

    existing = [] if force else _read_json_list(FILTER_RESULTS)
    existing_by_path = {
        canonical_path(item["path"]): item for item in existing if item.get("path")
    }

    results_by_path: dict[Path, dict[str, Any]] = {}
    pending: list[tuple[int, Path]] = []
    for index, wav in enumerate(slices, 1):
        key = canonical_path(wav)
        if key in existing_by_path:
            results_by_path[key] = existing_by_path[key]
        else:
            pending.append((index, wav))

    def ordered_results() -> list[dict[str, Any]]:
        return [
            results_by_path[canonical_path(wav)]
            for wav in slices
            if canonical_path(wav) in results_by_path
        ]

    def record_analysis(index: int, wav: Path, analysis_value: Any) -> None:
        analysis = _as_plain_dict(analysis_value)
        record = {"path": str(wav), **analysis}
        results_by_path[canonical_path(wav)] = record
        reason_suffix = f" ({record['reason']})" if record.get("reason") else ""
        print(
            f"  [{index}/{len(slices)}] {wav.name}  "
            f"{'保留' if record.get('keep') else '剔除'}"
            f"{reason_suffix}"
        )

    analyzed_count = 0
    worker_count = max(1, min(int(workers), len(pending) or 1))
    try:
        if worker_count > 1 and analyzer is analyze_slice:
            print(f"  使用 {worker_count} 个 CPU 进程并行提取音频特征")
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_analyze_slice_job, wav, thresholds): (index, wav)
                    for index, wav in pending
                }
                for future in as_completed(futures):
                    index, wav = futures[future]
                    _, analysis = future.result()
                    record_analysis(index, wav, analysis)
                    analyzed_count += 1
                    if analyzed_count % 25 == 0:
                        _write_json(FILTER_RESULTS, ordered_results())
        else:
            for index, wav in pending:
                record_analysis(index, wav, analyzer(wav, thresholds))
                analyzed_count += 1
    except BaseException:
        _write_json(FILTER_RESULTS, ordered_results())
        raise

    results = ordered_results()
    _write_json(FILTER_RESULTS, results)
    reasons = Counter(
        str(item.get("reason")) for item in results if not item.get("keep")
    )
    kept = sum(bool(item.get("keep")) for item in results)
    print(f"  ✓ 过滤标注已保存: {FILTER_RESULTS}")
    print(f"  本次分析 {analyzed_count} 段；累计保留 {kept}，剔除 {len(results) - kept}")
    if reasons:
        print("  剔除原因: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    return results


# ─────────────────────────── ⑥ Whisper 转写 ───────────────────────────


def _load_whisper_model(model_size: str) -> Any:
    import torch
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    requested_device = os.environ.get("HUAYIN_ASR_DEVICE", "auto")
    device = (
        requested_device
        if requested_device in {"cuda", "cpu"}
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_path = (
        model_size
        if "/" in model_size or os.path.isdir(model_size)
        else download_model(model_size)
    )
    print(f"加载 faster-whisper [{model_size}] → {model_path} on {device} ...")
    options: dict[str, Any] = {
        "device": device,
        "compute_type": "float16" if device == "cuda" else "int8",
    }
    cpu_threads = int(os.environ.get("HUAYIN_ASR_CPU_THREADS", "0"))
    if device == "cpu" and cpu_threads > 0:
        options["cpu_threads"] = cpu_threads
    return WhisperModel(model_path, **options)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def make_asr_record(path: Path, segments: Any, info: Any) -> dict[str, Any]:
    """Materialize Whisper output and retain its useful quality signals."""

    materialized = list(segments)
    text = "".join(str(getattr(segment, "text", "")) for segment in materialized).strip()
    lang = str(getattr(info, "language", None) or "").lower()

    weighted_logprob = 0.0
    logprob_weight = 0.0
    no_speech_values: list[float] = []
    compression_values: list[float] = []
    for segment in materialized:
        start = _finite_float(getattr(segment, "start", None))
        end = _finite_float(getattr(segment, "end", None))
        weight = max((end or 0.0) - (start or 0.0), 0.01)
        avg_logprob = _finite_float(getattr(segment, "avg_logprob", None))
        if avg_logprob is not None:
            weighted_logprob += avg_logprob * weight
            logprob_weight += weight
        no_speech = _finite_float(getattr(segment, "no_speech_prob", None))
        if no_speech is not None:
            no_speech_values.append(no_speech)
        compression = _finite_float(getattr(segment, "compression_ratio", None))
        if compression is not None:
            compression_values.append(compression)

    return {
        "path": str(path),
        "lang": lang,
        "text": text,
        "audio_duration": _wav_duration(path),
        "asr_quality_version": ASR_QUALITY_VERSION,
        "language_probability": _finite_float(
            getattr(info, "language_probability", None)
        ),
        "avg_logprob": (
            weighted_logprob / logprob_weight if logprob_weight else None
        ),
        "max_no_speech_probability": (
            max(no_speech_values) if no_speech_values else None
        ),
        "max_compression_ratio": (
            max(compression_values) if compression_values else None
        ),
        "segment_count": len(materialized),
    }


def has_asr_quality(item: Mapping[str, Any]) -> bool:
    audio_duration = _finite_float(item.get("audio_duration"))
    return (
        item.get("asr_quality_version") == ASR_QUALITY_VERSION
        and audio_duration is not None
        and audio_duration > 0
    )


def select_asr_slices(
    slices: list[Path],
    filter_results: list[dict[str, Any]],
    *,
    speaker_results: list[dict[str, Any]] | None = None,
    speaker_policy: str = "accept-or-uncertain",
) -> list[Path]:
    """Apply complete acoustic and optional three-state speaker annotations."""

    filter_by_path = {
        canonical_path(item["path"]): item
        for item in filter_results
        if item.get("path")
    }
    slice_keys = {canonical_path(path) for path in slices}
    missing_filters = slice_keys - set(filter_by_path)
    if missing_filters:
        raise SystemExit(
            f"✗ filter 结果缺少 {len(missing_filters)} 个白名单切片；"
            "请先补跑 filter 阶段"
        )
    acoustic_kept = [
        path
        for path in slices
        if filter_by_path[canonical_path(path)].get("keep") is True
    ]

    if speaker_results is None:
        if not SPEAKER_RESULTS.exists():
            return acoustic_kept
        speaker_results = _read_json_list(SPEAKER_RESULTS, required=True)
    speaker_by_path = {
        canonical_path(item["path"]): str(item.get("decision") or "")
        for item in speaker_results
        if item.get("path")
    }
    missing_speakers = {
        canonical_path(path)
        for path in acoustic_kept
        if canonical_path(path) not in speaker_by_path
    }
    if missing_speakers:
        raise SystemExit(
            f"✗ speaker 结果缺少 {len(missing_speakers)} 个声学保留切片；"
            "请先补跑 filter_speakers.py"
        )
    allowed_decisions = {
        "accept": {"accept"},
        "accept-or-uncertain": {"accept", "uncertain"},
    }.get(speaker_policy)
    if allowed_decisions is None:
        raise ValueError(f"unknown speaker policy: {speaker_policy}")
    return [
        path
        for path in acoustic_kept
        if speaker_by_path[canonical_path(path)] in allowed_decisions
    ]


def transcribe(
    model_size: str,
    *,
    model_loader: Callable[[str], Any] | None = None,
    speaker_policy: str = "accept-or-uncertain",
) -> list[dict[str, Any]]:
    all_slices = sorted(SLICES_DIR.glob("*.wav"))
    if not all_slices:
        raise SystemExit(f"✗ {SLICES_DIR} 下没有切片，请先跑 slice 阶段")
    slices = _select_training_derivatives(all_slices)
    if not slices:
        raise SystemExit("✗ 当前没有白名单素材对应的切片，请先跑 slice 阶段")

    filter_results = _read_json_list(FILTER_RESULTS, required=True)
    eligible = select_asr_slices(
        slices, filter_results, speaker_policy=speaker_policy
    )
    eligible_keys = {canonical_path(wav) for wav in eligible}

    cached = _read_json_list(ASR_RESULTS)
    results: list[dict[str, Any]] = []
    result_keys: set[Path] = set()
    for item in cached:
        if not item.get("path"):
            continue
        key = canonical_path(item["path"])
        if key not in result_keys:
            results.append(item)
            result_keys.add(key)

    pending = [wav for wav in eligible if canonical_path(wav) not in result_keys]
    if not pending:
        if results != cached or not ASR_RESULTS.exists():
            _write_json(ASR_RESULTS, results)
        print(f"  无新增待转写切片（已缓存 {len(results)} 条）")
        return results

    loader = _load_whisper_model if model_loader is None else model_loader
    model = loader(model_size)
    for index, wav in enumerate(pending, 1):
        segments, info = model.transcribe(
            str(wav),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            language=None,
        )
        item = make_asr_record(wav, segments, info)
        results.append(item)
        result_keys.add(canonical_path(wav))
        _write_json(ASR_RESULTS, results)
        print(
            f"  [{index}/{len(pending)}] {wav.name}  "
            f"({item['lang']})  {item['text'][:50]}"
        )

    print(f"  ✓ 转写结果已增量保存: {ASR_RESULTS}")
    return results


# ─────────────────────────── ⑦ 汇总训练集 ───────────────────────────


def is_repetitive_text(text: str, threshold: float = 0.5) -> bool:
    return text_quality.is_char_repetitive(text, threshold=threshold)


def build_dataset(
    transcripts: list[dict[str, Any]] | None = None,
    *,
    filter_results: list[dict[str, Any]] | None = None,
    speaker_results: list[dict[str, Any]] | None = None,
    alignment_results: list[dict[str, Any]] | None = None,
    require_alignment: bool = False,
    allow_en: bool = True,
    min_alignment_similarity: float = 0.45,
) -> Path:
    if transcripts is None:
        if not ASR_RESULTS.exists():
            raise SystemExit(f"✗ 没有 ASR 结果（{ASR_RESULTS}），请先跑 --stage asr")
        transcripts = _read_json_list(ASR_RESULTS, required=True)
        print(f"  从 {ASR_RESULTS} 载入 {len(transcripts)} 条转写结果")
    if filter_results is None:
        filter_results = _read_json_list(FILTER_RESULTS, required=True)

    filter_index = {
        canonical_path(item["path"]): item.get("keep") is True
        for item in filter_results
        if item.get("path")
    }
    if speaker_results is None and SPEAKER_RESULTS.exists():
        speaker_results = _read_json_list(SPEAKER_RESULTS, required=True)
    speaker_index = (
        {
            canonical_path(item["path"]): str(item.get("decision") or "")
            for item in speaker_results
            if item.get("path")
        }
        if speaker_results is not None
        else None
    )
    if alignment_results is None and ALIGNMENT_RESULTS.exists():
        alignment_results = _read_json_list(ALIGNMENT_RESULTS, required=False)
    alignment_index = (
        {
            canonical_path(item["path"]): item
            for item in alignment_results
            if item.get("path")
        }
        if alignment_results
        else {}
    )
    transcript_paths = {
        canonical_path(item["path"])
        for item in transcripts
        if item.get("path")
    }
    required_transcripts = {
        path
        for path, keep in filter_index.items()
        if keep and (speaker_index is None or speaker_index.get(path) == "accept")
    }
    missing_transcripts = required_transcripts - transcript_paths
    if missing_transcripts:
        raise SystemExit(
            f"✗ ASR 结果缺少 {len(missing_transcripts)} 个已保留切片；"
            "请先补跑 asr 阶段"
        )

    audio_out = DATASET_DIR / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped: Counter[str] = Counter()
    lines: list[str] = []
    output_names: set[str] = set()
    allowed_stems = _training_asset_stems()
    for item in transcripts:
        if not item.get("path"):
            skipped["missing_path"] += 1
            continue
        src = canonical_path(item["path"])
        if not _is_training_derivative(src, allowed_stems):
            skipped["source_whitelist"] += 1
            continue
        if filter_index.get(src) is not True:
            skipped["audio_filter"] += 1
            continue
        if speaker_index is not None:
            speaker_decision = speaker_index.get(src, "missing")
            if speaker_decision != "accept":
                skipped[f"speaker_{speaker_decision}"] += 1
                continue

        lang = str(item.get("lang") or "").lower()
        tag = LANG_TAG.get(lang)
        if tag is None:
            skipped["language"] += 1
            continue

        text = str(item.get("text") or "").replace("|", "/").strip()
        if len(text) < 2:
            skipped["short_text"] += 1
            continue
        text_gate = text_quality.evaluate_text_quality(
            text, lang, allow_en=allow_en
        )
        if not text_gate.keep:
            # Prefer the first reason for aggregate stats; full set is recoverable.
            skipped[f"text_{text_gate.reasons[0]}"] += 1
            continue
        if tag == "EN" and not allow_en:
            skipped["en_disabled"] += 1
            continue

        # Optional speech↔text alignment from forced-language re-ASR.
        align = alignment_index.get(src)
        if require_alignment and align is None:
            skipped["alignment_missing"] += 1
            continue
        if align is not None:
            if align.get("keep") is False:
                reason = str(align.get("reason") or "reject")
                skipped[f"alignment_{reason}"] += 1
                continue
            sim = _finite_float(align.get("similarity"))
            if sim is not None and sim < min_alignment_similarity:
                skipped["alignment_low_sim"] += 1
                continue
            # Prefer forced-language transcript when it is clearly better aligned
            # to the claimed language script and similarity is only moderate.
            forced_text = str(align.get("forced_text") or "").strip()
            if (
                forced_text
                and sim is not None
                and 0.45 <= sim < 0.75
                and text_quality.script_ratio(forced_text, lang)
                > text_quality.script_ratio(text, lang) + 0.15
            ):
                forced_gate = text_quality.evaluate_text_quality(
                    forced_text, lang, allow_en=allow_en
                )
                if forced_gate.keep:
                    text = forced_text.replace("|", "/")

        if has_asr_quality(item):
            audio_duration = _finite_float(item.get("audio_duration"))
            if audio_duration and audio_duration > 0:
                nonspace_length = len("".join(text.split()))
                if nonspace_length / audio_duration > 12.0:
                    skipped["text_density"] += 1
                    continue
            language_probability = _finite_float(item.get("language_probability"))
            avg_logprob = _finite_float(item.get("avg_logprob"))
            no_speech_probability = _finite_float(
                item.get("max_no_speech_probability")
            )
            compression_ratio = _finite_float(item.get("max_compression_ratio"))
            if (
                language_probability is not None
                and language_probability < MIN_LANGUAGE_PROBABILITY
            ):
                skipped["language_confidence"] += 1
                continue
            if avg_logprob is None or avg_logprob < MIN_AVG_LOGPROB:
                skipped["asr_logprob"] += 1
                continue
            if (
                no_speech_probability is not None
                and no_speech_probability > MAX_NO_SPEECH_PROBABILITY
            ):
                skipped["asr_no_speech"] += 1
                continue
            if (
                compression_ratio is not None
                and compression_ratio > MAX_COMPRESSION_RATIO
            ):
                skipped["asr_compression"] += 1
                continue
        if not src.is_file():
            skipped["missing_audio"] += 1
            continue
        if src.stat().st_size < 8_000:
            skipped["tiny_audio"] += 1
            continue

        kept += 1
        name = f"{kept:03d}_{tag.lower()}.wav"
        output_names.add(name)
        shutil.copy2(src, audio_out / name)
        lines.append(f"audio/{name}|花音|{tag}|{text}")

    # 删除本工具上次生成、但本次 annotation 已不再引用的编号文件。
    for old_audio in audio_out.glob("*.wav"):
        if GENERATED_AUDIO_RE.fullmatch(old_audio.name) and old_audio.name not in output_names:
            old_audio.unlink()

    list_path = DATASET_DIR / "annotation.list"
    payload = "\n".join(lines) + ("\n" if lines else "")
    tmp = list_path.with_name(f"{list_path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(list_path)

    skipped_total = sum(skipped.values())
    print(f"\n✓ 训练集就绪: {DATASET_DIR}")
    print(f"  保留 {kept} 段，丢弃 {skipped_total} 段")
    if skipped:
        print("  丢弃原因: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
    print(f"  标注文件: {list_path}")
    return list_path


# ─────────────────────────── 主入口 ───────────────────────────


def main() -> None:
    defaults = FilterThresholds()
    parser = argparse.ArgumentParser(description="花音训练数据管线")
    parser.add_argument(
        "--stage",
        choices=["all", "separate", "slice", "filter", "asr", "dataset"],
        default="all",
        help="单独运行某一阶段",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="faster-whisper 模型 size 或 HF repo id/本地路径",
    )
    parser.add_argument(
        "--speaker-policy",
        choices=["accept", "accept-or-uncertain"],
        default="accept-or-uncertain",
        help="ASR 声纹策略：只转 accept，或同时转 uncertain",
    )
    parser.add_argument(
        "--separation-chunk-minutes",
        type=float,
        default=SEPARATION_CHUNK_SECONDS / 60,
        help="人声分离长音频切块分钟数（默认 30，降低主存峰值）",
    )
    parser.add_argument(
        "--slice-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="slice 阶段并行 CPU 进程数（默认最多 4）",
    )
    parser.add_argument(
        "--filter-workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="filter 阶段并行 CPU 进程数（默认最多 12）",
    )
    parser.add_argument(
        "--voiced-ratio-threshold",
        type=float,
        default=defaults.voiced_ratio_threshold,
    )
    parser.add_argument(
        "--f0-cv-threshold", type=float, default=defaults.f0_cv_threshold
    )
    parser.add_argument(
        "--longest-voiced-threshold",
        type=float,
        default=defaults.longest_voiced_threshold,
    )
    parser.add_argument("--min-rms-db", type=float, default=defaults.min_rms_db)
    parser.add_argument("--min-duration", type=float, default=defaults.min_duration)
    parser.add_argument("--max-duration", type=float, default=defaults.max_duration)
    parser.add_argument("--fmin", type=float, default=defaults.fmin)
    parser.add_argument("--fmax", type=float, default=defaults.fmax)
    parser.add_argument(
        "--pitch-hop-length", type=int, default=defaults.hop_length
    )
    parser.add_argument(
        "--force-refilter",
        action="store_true",
        help="忽略 filter_results.json，按当前阈值重新分析全部切片",
    )
    parser.add_argument(
        "--allow-en",
        action="store_true",
        help="dataset 阶段保留英语条目（默认仅中日，保证 TTS 纯度）",
    )
    parser.add_argument(
        "--require-alignment",
        action="store_true",
        help="dataset 阶段要求 alignment_results.json 校验通过",
    )
    parser.add_argument(
        "--min-alignment-similarity",
        type=float,
        default=0.45,
        help="音文对齐最低相似度（配合 --require-alignment）",
    )
    args = parser.parse_args()

    thresholds = FilterThresholds(
        voiced_ratio_threshold=args.voiced_ratio_threshold,
        f0_cv_threshold=args.f0_cv_threshold,
        longest_voiced_threshold=args.longest_voiced_threshold,
        min_rms_db=args.min_rms_db,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_length=args.pitch_hop_length,
    )
    if thresholds.min_duration > thresholds.max_duration:
        parser.error("--min-duration 不能大于 --max-duration")
    if thresholds.fmin >= thresholds.fmax:
        parser.error("--fmin 必须小于 --fmax")
    if args.slice_workers < 1:
        parser.error("--slice-workers 必须至少为 1")
    if args.filter_workers < 1:
        parser.error("--filter-workers 必须至少为 1")
    if args.separation_chunk_minutes <= 0:
        parser.error("--separation-chunk-minutes 必须大于 0")

    print(f"=== 花音训练数据管线 | stage={args.stage} ===\n")

    if args.stage in ("all", "separate"):
        print("③ 人声分离（白名单 + UVR5 BS-Roformer）")
        separate_vocals(
            chunk_duration_seconds=max(
                1, int(args.separation_chunk_minutes * 60)
            )
        )
        print()

    if args.stage in ("all", "slice"):
        print("④ 静音切分（逐 vocal 增量）")
        slice_audio(workers=args.slice_workers)
        print()

    if args.stage in ("all", "filter"):
        print("⑤ 音频特征过滤（歌声 / 能量 / 时长）")
        filter_slices(
            thresholds,
            force=args.force_refilter,
            workers=args.filter_workers,
        )
        print()

    transcripts: list[dict[str, Any]] | None = None
    if args.stage in ("all", "asr"):
        print("⑥ Whisper 增量转写 + 语种识别")
        transcripts = transcribe(
            args.whisper_model, speaker_policy=args.speaker_policy
        )
        print()

    if args.stage in ("all", "dataset"):
        print("⑦ 汇总训练集（语言 / 文本质量 / 可选音文对齐）")
        build_dataset(
            transcripts,
            allow_en=args.allow_en,
            require_alignment=args.require_alignment,
            min_alignment_similarity=args.min_alignment_similarity,
        )


if __name__ == "__main__":
    main()
