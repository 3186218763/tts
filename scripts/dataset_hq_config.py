"""Load HQ dataset thresholds from YAML into frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ScoreWeights:
    align: float
    asr: float
    spk: float
    text: float
    src: float
    dur: float


@dataclass(frozen=True)
class ValidationConfig:
    fail_on_zh_shortfall: bool
    align_median_min: float
    align_p10_min: float
    speaker_p5_margin: float
    listen_ok_rate: float
    listen_l2_ok_rate: float
    listen_l3_ok_rate: float
    listen_critical_fail_rate: float


@dataclass(frozen=True)
class DatasetHQConfig:
    n_target: int
    zh_ratio: float
    s_min: float
    t_align: float
    t_align_clip: float
    max_source_share: float
    duration_min: float
    duration_max: float
    min_rms_db: float
    voiced_ratio_threshold: float
    f0_cv_threshold: float
    longest_voiced_threshold: float
    speaker_policy: str
    clip_min_score_lower: float
    min_language_probability: float
    min_avg_logprob: float
    max_no_speech_probability: float
    max_compression_ratio: float
    listen_sample_size: int
    est_zh_clips_per_hour_b: float
    est_zh_clips_per_hour_a: float
    weights: ScoreWeights
    tier_src_score: dict[str, float]
    min_duration_by_tier: dict[str, int]
    validation: ValidationConfig


def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"dataset_hq config missing key: {key}")
    return data[key]


def load_dataset_hq_config(path: Path) -> DatasetHQConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    weights_raw = _require(raw, "weights")
    weights = ScoreWeights(
        align=float(weights_raw["align"]),
        asr=float(weights_raw["asr"]),
        spk=float(weights_raw["spk"]),
        text=float(weights_raw["text"]),
        src=float(weights_raw["src"]),
        dur=float(weights_raw["dur"]),
    )

    val_raw = _require(raw, "validation")
    validation = ValidationConfig(
        fail_on_zh_shortfall=bool(val_raw["fail_on_zh_shortfall"]),
        align_median_min=float(val_raw["align_median_min"]),
        align_p10_min=float(val_raw["align_p10_min"]),
        speaker_p5_margin=float(val_raw["speaker_p5_margin"]),
        listen_ok_rate=float(val_raw["listen_ok_rate"]),
        listen_l2_ok_rate=float(val_raw["listen_l2_ok_rate"]),
        listen_l3_ok_rate=float(val_raw["listen_l3_ok_rate"]),
        listen_critical_fail_rate=float(val_raw["listen_critical_fail_rate"]),
    )

    tier_src = {str(k): float(v) for k, v in dict(_require(raw, "tier_src_score")).items()}
    min_dur = {str(k): int(v) for k, v in dict(_require(raw, "min_duration_by_tier")).items()}

    return DatasetHQConfig(
        n_target=int(_require(raw, "n_target")),
        zh_ratio=float(_require(raw, "zh_ratio")),
        s_min=float(_require(raw, "s_min")),
        t_align=float(_require(raw, "t_align")),
        t_align_clip=float(_require(raw, "t_align_clip")),
        max_source_share=float(_require(raw, "max_source_share")),
        duration_min=float(_require(raw, "duration_min")),
        duration_max=float(_require(raw, "duration_max")),
        min_rms_db=float(_require(raw, "min_rms_db")),
        voiced_ratio_threshold=float(_require(raw, "voiced_ratio_threshold")),
        f0_cv_threshold=float(_require(raw, "f0_cv_threshold")),
        longest_voiced_threshold=float(_require(raw, "longest_voiced_threshold")),
        speaker_policy=str(_require(raw, "speaker_policy")),
        clip_min_score_lower=float(_require(raw, "clip_min_score_lower")),
        min_language_probability=float(_require(raw, "min_language_probability")),
        min_avg_logprob=float(_require(raw, "min_avg_logprob")),
        max_no_speech_probability=float(_require(raw, "max_no_speech_probability")),
        max_compression_ratio=float(_require(raw, "max_compression_ratio")),
        listen_sample_size=int(_require(raw, "listen_sample_size")),
        est_zh_clips_per_hour_b=float(_require(raw, "est_zh_clips_per_hour_b")),
        est_zh_clips_per_hour_a=float(_require(raw, "est_zh_clips_per_hour_a")),
        weights=weights,
        tier_src_score=tier_src,
        min_duration_by_tier=min_dur,
        validation=validation,
    )


def config_to_dict(cfg: DatasetHQConfig) -> dict[str, Any]:
    """JSON-serializable snapshot of thresholds."""
    from dataclasses import asdict

    return asdict(cfg)
