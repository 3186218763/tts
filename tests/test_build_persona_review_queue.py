from __future__ import annotations

import json
from pathlib import Path

import pytest

# 脚本在 scripts/ 下，测试里按项目惯例 import
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_persona_review_queue as bq  # noqa: E402


def test_normalize_text_collapses_space():
    assert bq.normalize_text("  阿里  嘎多  ") == "阿里嘎多"


def test_content_len_gate():
    assert bq.in_length_range("哈哈") is False  # 过短
    assert bq.in_length_range("这可是笨蛋尼们") is True


def test_guess_era_from_source_peak():
    assert bq.guess_era_tag("BV1xx_2022-05-01_杂谈.wav") == "peak"
    assert bq.guess_era_tag("【眞白花音】2026_04_15 白菜来啦.wav") == "late"


def test_build_queue_filters_and_caps(tmp_path: Path):
    asr = [
        {
            "path": str(tmp_path / "BV_2022_peak_(Vocals)__1.wav"),
            "lang": "zh",
            "text": "尼们现在是什么心情呀",
            "avg_logprob": -0.1,
            "language_probability": 0.95,
            "segment_count": 1,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.0,
        },
        {
            "path": str(tmp_path / "BV_2026_late_(Vocals)__2.wav"),
            "lang": "zh",
            "text": "谢谢大家今天也来看直播",
            "avg_logprob": -0.1,
            "language_probability": 0.95,
            "segment_count": 1,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.0,
        },
        {
            "path": str(tmp_path / "bad.wav"),
            "lang": "zh",
            "text": "啊",
            "avg_logprob": -0.1,
            "language_probability": 0.95,
            "segment_count": 1,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.0,
        },
    ]
    # 写假 asr 文件
    asr_path = tmp_path / "asr.json"
    asr_path.write_text(json.dumps(asr, ensure_ascii=False), encoding="utf-8")
    seed_fewshot = {
        "asr_pool": [
            {
                "text": "摆在不能原谅披萨汉堡里面的菠萝",
                "lang": "zh",
                "avg_logprob": -0.15,
                "high_conf": True,
                "scenes": ["自嘲"],
                "source": "seed.wav",
            }
        ]
    }
    few_path = tmp_path / "few.json"
    few_path.write_text(json.dumps(seed_fewshot, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "queue.json"
    stats = bq.build_queue(
        asr_path=asr_path,
        fewshot_path=few_path,
        out_path=out,
        max_items=50,
        prefer_zh=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert stats["written"] >= 1
    texts = {item["text"] for item in data["items"]}
    assert "尼们现在是什么心情呀" in texts
    assert "啊" not in texts
    for item in data["items"]:
        assert item["status"] == "pending"
        assert item["era_tag"] in {"peak", "late", "unknown"}
