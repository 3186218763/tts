import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataclasses import replace

from dataset_hq_config import ValidationConfig, load_dataset_hq_config
from validate_hq_lib import (
    run_auto_validation,
    build_listen_sample,
    evaluate_listen_results,
)

CFG = load_dataset_hq_config(ROOT / "configs" / "dataset_hq.yaml")


def _sc(
    path,
    decision="candidate",
    score=80,
    lang="zh",
    sim=0.9,
    spk_l=0.4,
    text="大家好啊",
    tier="B",
    bvid="BVa",
):
    return {
        "path": path,
        "decision": decision,
        "score": score,
        "lang": lang,
        "text": text,
        "tier": tier,
        "source_bvid": bvid,
        "metrics": {"similarity": sim, "score_lower": spk_l},
    }


def test_auto_pass_minimal():
    manifest = [
        {
            "path": "/s/a.wav",
            "dataset_path": "audio/00001_zh.wav",
            "lang": "zh",
            "text": "大家好啊",
            "source_bvid": "BVa",
            "score": 80,
            "tier": "B",
        }
    ]
    stats = {"n_zh": 1, "n_ja": 0, "n_total": 1, "zh_shortfall": False}
    sc = {"/s/a.wav": _sc("/s/a.wav")}
    report = run_auto_validation(
        manifest,
        stats,
        sc,
        annotation_lines=["audio/00001_zh.wav|花音|ZH|大家好啊"],
        wav_names=["00001_zh.wav"],
        cfg=CFG,
    )
    assert report["auto_pass"] is True
    assert report["verdict"] == "pending_listen"


def test_fail_v2_zh_ratio():
    manifest = [
        {
            "path": f"/s/{i}.wav",
            "dataset_path": f"audio/{i:05d}_jp.wav",
            "lang": "ja",
            "text": f"あいう{i}",
            "source_bvid": f"B{i}",
            "score": 80,
            "tier": "C",
        }
        for i in range(10)
    ]
    stats = {"n_zh": 2, "n_ja": 8, "n_total": 10, "zh_shortfall": False}
    sc = {
        m["path"]: _sc(m["path"], lang="ja", text=m["text"], bvid=m["source_bvid"])
        for m in manifest
    }
    report = run_auto_validation(
        manifest,
        stats,
        sc,
        annotation_lines=[f"audio/{i:05d}_jp.wav|花音|JP|あいう{i}" for i in range(10)],
        wav_names=[f"{i:05d}_jp.wav" for i in range(10)],
        cfg=CFG,
    )
    assert report["auto_pass"] is False
    assert any(c["id"] == "V2" and not c["pass"] for c in report["checks"])


def _strict_cfg():
    v = CFG.validation
    return replace(
        CFG,
        validation=ValidationConfig(
            fail_on_zh_shortfall=True,
            align_median_min=v.align_median_min,
            align_p10_min=v.align_p10_min,
            speaker_p5_margin=v.speaker_p5_margin,
            listen_ok_rate=v.listen_ok_rate,
            listen_l2_ok_rate=v.listen_l2_ok_rate,
            listen_l3_ok_rate=v.listen_l3_ok_rate,
            listen_critical_fail_rate=v.listen_critical_fail_rate,
        ),
    )


def test_v1_shortfall_tolerated_by_default():
    stats = {"n_zh": 10, "n_ja": 6, "n_total": 16, "zh_shortfall": True, "n_target": 20}
    report = run_auto_validation([], stats, {}, [], [], CFG)
    v1 = [c for c in report["checks"] if c["id"] == "V1"][0]
    assert v1["pass"] is True
    assert any("zh_shortfall" in w for w in report["warnings"])


def test_v1_shortfall_fails_when_strict():
    stats = {"n_zh": 10, "n_ja": 6, "n_total": 16, "zh_shortfall": True}
    report = run_auto_validation([], stats, {}, [], [], _strict_cfg())
    assert any(c["id"] == "V1" and not c["pass"] for c in report["checks"])


def test_fail_v3_not_candidate():
    manifest = [
        {
            "path": "/s/a.wav",
            "dataset_path": "audio/00001_zh.wav",
            "lang": "zh",
            "text": "大家好啊",
            "source_bvid": "BVa",
            "score": 80,
            "tier": "B",
        }
    ]
    stats = {"n_zh": 1, "n_ja": 0, "n_total": 1, "zh_shortfall": False}
    sc = {"/s/a.wav": _sc("/s/a.wav", decision="reject")}
    report = run_auto_validation(
        manifest,
        stats,
        sc,
        ["audio/00001_zh.wav|花音|ZH|大家好啊"],
        ["00001_zh.wav"],
        CFG,
    )
    assert any(c["id"] == "V3" and not c["pass"] for c in report["checks"])


def test_fail_v8_count_mismatch():
    manifest = [
        {
            "path": "/s/a.wav",
            "dataset_path": "audio/00001_zh.wav",
            "lang": "zh",
            "text": "大家好啊",
            "source_bvid": "BVa",
            "score": 80,
            "tier": "B",
        }
    ]
    stats = {"n_zh": 1, "n_ja": 0, "n_total": 1, "zh_shortfall": False}
    sc = {"/s/a.wav": _sc("/s/a.wav")}
    report = run_auto_validation(
        manifest,
        stats,
        sc,
        ["audio/00001_zh.wav|花音|ZH|大家好啊"],
        ["00001_zh.wav", "extra.wav"],
        CFG,
    )
    assert any(c["id"] == "V8" and not c["pass"] for c in report["checks"])


def test_stratified_sample_layers():
    manifest = []
    sc = {}
    for i in range(30):
        p = f"/B/b{i}.wav"
        manifest.append(
            {
                "path": p,
                "lang": "zh",
                "text": f"中文句子{i}",
                "tier": "B",
                "source_bvid": "BVb",
                "score": 80,
                "dataset_path": f"a{i}.wav",
            }
        )
        sc[p] = _sc(p, tier="B", text=f"中文句子{i}", sim=0.9)
    for i in range(30):
        p = f"/A/a{i}.wav"
        manifest.append(
            {
                "path": p,
                "lang": "zh",
                "text": f"切片句子{i}",
                "tier": "A",
                "source_bvid": "BVa",
                "score": 70,
                "dataset_path": f"c{i}.wav",
            }
        )
        sc[p] = _sc(p, tier="A", text=f"切片句子{i}", sim=0.48)
    for i in range(20):
        p = f"/J/j{i}.wav"
        manifest.append(
            {
                "path": p,
                "lang": "ja",
                "text": f"日本語{i}",
                "tier": "C",
                "source_bvid": "BVj",
                "score": 75,
                "dataset_path": f"j{i}.wav",
            }
        )
        sc[p] = _sc(p, lang="ja", tier="C", text=f"日本語{i}", sim=0.9)
    sample = build_listen_sample(manifest, sc, n=80, seed=42, cfg=CFG)
    layers = {x["layer"] for x in sample}
    assert "L1" in layers and "L2" in layers and "L5" in layers


def test_listen_ok_rate_and_critical():
    rows = [{"id": f"i{n}", "layer": "L1", "label": "OK"} for n in range(17)]
    rows += [{"id": "x", "layer": "L2", "label": "非本人"}]
    rows += [{"id": f"j{n}", "layer": "L2", "label": "OK"} for n in range(10)]
    result = evaluate_listen_results(rows, CFG)
    assert "ok_rate" in result
    assert "pass" in result
    # L2 has 1/11 非本人 ~9% < 10% critical; overall mostly OK
    # 1 critical in L2 of 11 = 0.0909 < 0.10, so may pass critical
    assert result["ok_rate"] > 0.8
