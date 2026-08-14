import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_hq_config import load_dataset_hq_config


def test_load_default_config_has_spec_defaults():
    cfg = load_dataset_hq_config(ROOT / "configs" / "dataset_hq.yaml")
    assert cfg.n_target == 20000
    assert cfg.zh_ratio == 0.60
    assert cfg.s_min == 55.0
    assert cfg.t_align == 0.45
    assert cfg.t_align_clip == 0.50
    assert cfg.max_source_share == 0.15
    assert cfg.duration_min == 1.5
    assert cfg.duration_max == 20.0
    assert cfg.min_rms_db == -40.0
    assert cfg.speaker_policy == "accept"
    assert cfg.clip_min_score_lower == 0.25
    assert cfg.weights.align == 30
    assert cfg.weights.asr == 20
    assert cfg.weights.spk == 20
    assert cfg.weights.text == 15
    assert cfg.weights.src == 10
    assert cfg.weights.dur == 5
    assert cfg.tier_src_score["B"] == 1.0
    assert cfg.tier_src_score["A"] == 0.75
    assert cfg.tier_src_score["C"] == 0.6
    assert cfg.min_duration_by_tier["A"] == 30
    assert cfg.validation.fail_on_zh_shortfall is False
    assert cfg.validation.align_median_min == 0.55
    assert cfg.validation.align_p10_min == 0.45
    assert cfg.est_zh_clips_per_hour_b == 80.0
    assert cfg.est_zh_clips_per_hour_a == 200.0
