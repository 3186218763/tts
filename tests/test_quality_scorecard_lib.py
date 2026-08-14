import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_hq_config import load_dataset_hq_config
from quality_scorecard_lib import score_one, build_scorecard_items, extract_bvid

CFG = load_dataset_hq_config(ROOT / "configs" / "dataset_hq.yaml")


def _green(
    tier="B",
    sim=0.95,
    dur=4.0,
    spk_lower=0.4,
    lang="zh",
    text="大家好今天也来直播啦",
):
    path = "/data/slices/BV14qDyBYEDy_title_(Vocals)__0_1.wav"
    return score_one(
        path=path,
        filter_row={
            "path": path,
            "keep": True,
            "reason": "keep",
            "duration": dur,
            "rms_db": -20.0,
            "voiced_ratio": 0.3,
            "f0_cv": 0.3,
            "longest_voiced": 0.5,
        },
        speaker_row={
            "path": path,
            "decision": "accept",
            "score": 0.55,
            "score_lower": spk_lower,
        },
        asr_row={
            "path": path,
            "lang": lang,
            "text": text,
            "language_probability": 0.95,
            "avg_logprob": -0.3,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.2,
            "audio_duration": dur,
            "asr_quality_version": 1,
        },
        align_row={
            "path": path,
            "similarity": sim,
            "keep": True,
            "forced_text": text,
            "original_text": text,
        },
        tier=tier,
        risks=[],
        cfg=CFG,
    )


def test_extract_bvid():
    assert (
        extract_bvid("/x/BV14qDyBYEDy_【眞白花音】a_(Vocals)__1_2.wav")
        == "BV14qDyBYEDy"
    )


def test_hard_reject_duration_even_if_filter_keep():
    item = score_one(
        path="/data/slices/BV14qDyBYEDy_t_(Vocals)__0_1.wav",
        filter_row={
            "keep": True,
            "reason": "keep",
            "duration": 25.0,
            "rms_db": -20,
            "voiced_ratio": 0.3,
            "f0_cv": 0.3,
            "longest_voiced": 0.5,
        },
        speaker_row={"decision": "accept", "score": 0.5, "score_lower": 0.4},
        asr_row={
            "lang": "zh",
            "text": "大家好今天也来直播啦",
            "language_probability": 0.9,
            "avg_logprob": -0.3,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.2,
            "audio_duration": 25.0,
            "asr_quality_version": 1,
        },
        align_row={
            "similarity": 0.9,
            "keep": True,
            "forced_text": "大家好今天也来直播啦",
        },
        tier="B",
        risks=[],
        cfg=CFG,
    )
    assert item["decision"] == "reject"
    assert "duration" in item["reject_reasons"]


def test_hard_reject_singing():
    item = score_one(
        path="/data/slices/BV14qDyBYEDy_t_(Vocals)__0_1.wav",
        filter_row={
            "keep": False,
            "reason": "singing",
            "duration": 4.0,
            "rms_db": -20,
            "voiced_ratio": 0.9,
            "f0_cv": 0.05,
            "longest_voiced": 5.0,
        },
        speaker_row={"decision": "accept", "score": 0.5, "score_lower": 0.4},
        asr_row={
            "lang": "zh",
            "text": "啦啦啦歌唱今天",
            "language_probability": 0.9,
            "avg_logprob": -0.3,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.2,
            "audio_duration": 4.0,
            "asr_quality_version": 1,
        },
        align_row={"similarity": 0.9, "keep": True, "forced_text": "啦啦啦歌唱今天"},
        tier="B",
        risks=[],
        cfg=CFG,
    )
    assert item["decision"] == "reject"
    assert "singing" in item["reject_reasons"]


def test_missing_asr_rejects_without_raise():
    path = "/data/slices/BV14qDyBYEDy_t_(Vocals)__0_1.wav"
    item = score_one(
        path=path,
        filter_row={
            "keep": True,
            "duration": 4.0,
            "rms_db": -20,
            "voiced_ratio": 0.3,
            "f0_cv": 0.3,
            "longest_voiced": 0.5,
        },
        speaker_row={"decision": "accept", "score": 0.5, "score_lower": 0.4},
        asr_row=None,
        align_row=None,
        tier="B",
        risks=[],
        cfg=CFG,
    )
    assert item["decision"] == "reject"
    assert "missing_asr" in item["reject_reasons"]
    assert "missing_alignment" in item["reject_reasons"]


def test_candidate_zh_gets_score():
    item = _green()
    assert item["decision"] == "candidate"
    assert 0 < item["score"] <= 100
    assert item["lang"] == "zh"
    assert item["text"] == "大家好今天也来直播啦"


def test_clip_stricter_align():
    c = _green(tier="C", sim=0.47)
    a = _green(tier="A", sim=0.47, spk_lower=0.4)
    assert "alignment" not in c["reject_reasons"]
    assert a["decision"] == "reject"
    assert "alignment" in a["reject_reasons"]


def test_clip_stricter_speaker_score_lower():
    a = _green(tier="A", sim=0.95, spk_lower=0.10)
    c = _green(tier="C", sim=0.95, spk_lower=0.10)
    assert a["decision"] == "reject"
    assert "speaker" in a["reject_reasons"]
    assert "speaker" not in c["reject_reasons"]


def test_borderline_or_candidate_low_components():
    path = "/data/slices/BV14qDyBYEDy_t_(Vocals)__0_1.wav"
    item = score_one(
        path=path,
        filter_row={
            "keep": True,
            "duration": 4.0,
            "rms_db": -20,
            "voiced_ratio": 0.3,
            "f0_cv": 0.3,
            "longest_voiced": 0.5,
        },
        speaker_row={"decision": "accept", "score": 0.2, "score_lower": 0.2},
        asr_row={
            "lang": "zh",
            "text": "大家好今天也来直播啦",
            "language_probability": 0.51,
            "avg_logprob": -0.99,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.2,
            "audio_duration": 4.0,
            "asr_quality_version": 1,
        },
        align_row={
            "similarity": 0.46,
            "keep": True,
            "forced_text": "大家好今天也来直播啦",
        },
        tier="C",
        risks=["watch"],
        cfg=CFG,
    )
    assert item["decision"] in ("borderline", "candidate")
    if item["score"] < CFG.s_min:
        assert item["decision"] == "borderline"


def test_align_improves_score_monotone():
    low = _green(sim=0.5)
    high = _green(sim=0.99)
    assert high["score"] >= low["score"]


def test_build_scorecard_union_paths():
    items = build_scorecard_items(
        filter_rows=[
            {
                "path": "/a.wav",
                "keep": True,
                "duration": 3,
                "rms_db": -20,
                "voiced_ratio": 0.2,
                "f0_cv": 0.2,
                "longest_voiced": 0.2,
            }
        ],
        speaker_rows=[],
        asr_rows=[],
        align_rows=[],
        bvid_to_meta={},
        cfg=CFG,
        canonical_path=lambda p: p,
    )
    assert len(items) == 1
    assert items[0]["decision"] == "reject"


def test_zh_text_forced_to_simplified():
    item = _green(text="我現在也在接維修師因為擺在就是我的維修師用")
    assert item["text"] == "我现在也在接维修师因为摆在就是我的维修师用"
    assert item["decision"] == "candidate"


def test_ja_text_not_simplified():
    item = _green(lang="ja", text="雑談してますけどね")
    assert item["text"] == "雑談してますけどね"
    assert item["decision"] == "candidate"
