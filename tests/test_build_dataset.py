import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import build_dataset as bd


def _feature_values(**overrides):
    values = {
        "duration": 5.0,
        "rms_db": -20.0,
        "voiced_ratio": 0.50,
        "f0_cv": 0.30,
        "longest_voiced": 1.0,
    }
    values.update(overrides)
    return values


def _field(value, name):
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _decision(value):
    if isinstance(value, tuple):
        return value
    return _field(value, "keep"), _field(value, "reason")


def _json_ready(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value.__dict__


def _write_large_audio(path: Path, size: int = 9_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + b"\0" * (size - 4))
    return path


def _resolved_from_project(path, project_root: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = project_root / value
    return value.resolve()


@pytest.fixture
def pipeline_paths(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    paths = SimpleNamespace(
        root=tmp_path,
        wav=data_dir / "wav",
        vocals=data_dir / "vocals",
        slices=data_dir / "slices",
        dataset=data_dir / "dataset",
        assets=data_dir / "training_assets.txt",
        filters=data_dir / "filter_results.json",
        speakers=data_dir / "speaker_results.json",
        asr=data_dir / "asr_results.json",
    )
    for directory in (paths.wav, paths.vocals, paths.slices, paths.dataset):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bd, "PROJECT_ROOT", paths.root)
    monkeypatch.setattr(bd, "WAV_DIR", paths.wav)
    monkeypatch.setattr(bd, "VOCALS_DIR", paths.vocals)
    monkeypatch.setattr(bd, "SLICES_DIR", paths.slices)
    monkeypatch.setattr(bd, "DATASET_DIR", paths.dataset)
    monkeypatch.setattr(bd, "TRAINING_ASSETS", paths.assets, raising=False)
    monkeypatch.setattr(bd, "FILTER_RESULTS", paths.filters, raising=False)
    monkeypatch.setattr(bd, "SPEAKER_RESULTS", paths.speakers, raising=False)
    monkeypatch.setattr(bd, "ASR_RESULTS", paths.asr)
    return paths


# ---------------------------------------------------------------------------
# Whitelist and filename matching


def test_load_training_assets_returns_only_listed_wavs_and_ignores_blank_lines(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    first = wav_dir / "[眞白花音_录播]第一场.wav"
    second = wav_dir / "第二场.wav"
    unlisted = wav_dir / "不要处理.wav"
    for path in (first, second, unlisted):
        path.touch()

    assets = tmp_path / "training_assets.txt"
    assets.write_text(f"  {first.name}  \n\n{second.name}\n", encoding="utf-8")

    assert bd.load_training_assets(assets, wav_dir) == [first, second]


def test_load_training_assets_fails_clearly_when_list_is_missing(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        bd.load_training_assets(tmp_path / "missing.txt", tmp_path / "wav")


def test_load_training_assets_fails_when_a_listed_wav_is_missing(tmp_path):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    assets = tmp_path / "training_assets.txt"
    assets.write_text("missing.wav\n", encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        bd.load_training_assets(assets, wav_dir)


def test_has_output_with_prefix_treats_square_brackets_as_literal(tmp_path):
    output = tmp_path / "[眞白花音_录播]第一场_(Vocals)_model.wav"
    output.touch()

    assert bd._has_output_with_prefix(
        tmp_path, "[眞白花音_录播]第一场", marker="_(Vocals)"
    )
    assert not bd._has_output_with_prefix(
        tmp_path, "[眞白花音_录播]第二场", marker="_(Vocals)"
    )


def test_separate_vocals_only_processes_pending_whitelisted_files(
    pipeline_paths, monkeypatch
):
    completed = pipeline_paths.wav / "[眞白花音_录播]已完成.wav"
    pending = pipeline_paths.wav / "应处理.wav"
    unlisted = pipeline_paths.wav / "不应处理.wav"
    for source in (completed, pending, unlisted):
        source.touch()
    pipeline_paths.assets.write_text(
        f"{completed.name}\n{pending.name}\n", encoding="utf-8"
    )
    (pipeline_paths.vocals / f"{completed.stem}_(Vocals)_model.wav").touch()

    separated = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_model(self, model):
            self.model = model

        def separate(self, path):
            separated.append(Path(path))
            return [f"{Path(path).stem}_(Vocals)_model.wav"]

    fake_separator_module = ModuleType("audio_separator.separator")
    fake_separator_module.Separator = FakeSeparator
    fake_package = ModuleType("audio_separator")
    fake_package.separator = fake_separator_module
    monkeypatch.setitem(sys.modules, "audio_separator", fake_package)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", fake_separator_module)

    bd.separate_vocals()

    assert separated == [pending]


def test_slice_audio_skips_completed_bracketed_vocal_but_processes_new_one(
    pipeline_paths, monkeypatch
):
    completed = pipeline_paths.vocals / "[眞白花音_录播]旧场_(Vocals)_model.wav"
    pending = pipeline_paths.vocals / "[眞白花音_录播]新场_(Vocals)_model.wav"
    completed.touch()
    pending.touch()
    (pipeline_paths.slices / f"{completed.stem}__0000000000_0000003200.wav").touch()

    loaded = []

    def fake_load_audio(path, sr):
        loaded.append((Path(path), sr))
        return np.ones(3_200, dtype=np.float32)

    class FakeSlicer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def slice(self, audio):
            return [(audio, 0, len(audio))]

    fake_wavfile = ModuleType("scipy.io.wavfile")
    fake_wavfile.write = Mock()
    fake_scipy_io = ModuleType("scipy.io")
    fake_scipy_io.wavfile = fake_wavfile
    fake_scipy = ModuleType("scipy")
    fake_scipy.io = fake_scipy_io
    fake_slicer2 = ModuleType("slicer2")
    fake_slicer2.Slicer = FakeSlicer

    monkeypatch.setattr(bd, "load_audio", fake_load_audio)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy_io)
    monkeypatch.setitem(sys.modules, "scipy.io.wavfile", fake_wavfile)
    monkeypatch.setitem(sys.modules, "slicer2", fake_slicer2)

    bd.slice_audio()

    assert loaded == [(pending, 32_000)]
    assert fake_wavfile.write.call_count == 1
    written_path = Path(fake_wavfile.write.call_args.args[0])
    assert written_path.name.startswith(f"{pending.stem}__")


# ---------------------------------------------------------------------------
# Audio feature extraction and pure classification


@pytest.mark.parametrize(
    ("features", "reason"),
    [
        (_feature_values(voiced_ratio=0.76, f0_cv=0.14), "singing"),
        (_feature_values(longest_voiced=3.01), "singing"),
        (_feature_values(rms_db=-40.01), "low_energy"),
        (_feature_values(duration=1.49), "duration"),
        (_feature_values(duration=30.01), "duration"),
    ],
)
def test_classify_audio_features_rejects_bad_audio(features, reason):
    keep, actual_reason = _decision(
        bd.classify_audio_features(features, bd.FilterThresholds())
    )

    assert keep is False
    if reason == "duration":
        assert actual_reason in {"duration", "too_short", "too_long"}
    else:
        assert actual_reason == reason


def test_classify_audio_features_keeps_normal_speech():
    assert _decision(
        bd.classify_audio_features(_feature_values(), bd.FilterThresholds())
    ) == (True, None)


@pytest.mark.parametrize(
    "features",
    [
        _feature_values(voiced_ratio=0.75, f0_cv=0.14),
        _feature_values(voiced_ratio=0.76, f0_cv=0.15),
        _feature_values(longest_voiced=3.0),
        _feature_values(rms_db=-40.0),
        _feature_values(duration=1.5),
        _feature_values(duration=30.0),
    ],
)
def test_classify_audio_features_uses_strict_thresholds(features):
    assert _decision(
        bd.classify_audio_features(features, bd.FilterThresholds())
    ) == (True, None)


def test_extract_audio_features_uses_longest_voiced_run_and_fixed_dbfs_reference():
    sr = 1_024
    hop_length = 512
    audio = np.full(sr * 4, 0.1, dtype=np.float32)
    f0 = np.array([300, 100, np.nan, 100, 150, 200, 250, np.nan], dtype=float)
    voiced = np.array([True, True, False, True, True, True, True, False])
    captured = {}

    def fake_pyin(*args, **kwargs):
        captured.update(kwargs)
        supplied_audio = args[0] if args else kwargs["y"]
        assert np.array_equal(supplied_audio, audio)
        return f0, voiced, np.ones_like(f0)

    features = bd.extract_audio_features(
        audio,
        sr,
        pyin_fn=fake_pyin,
        hop_length=hop_length,
        fmin=80,
        fmax=400,
    )

    assert _field(features, "duration") == pytest.approx(4.0)
    assert _field(features, "rms_db") == pytest.approx(-20.0, abs=0.2)
    assert _field(features, "voiced_ratio") == pytest.approx(6 / 8)
    assert _field(features, "longest_voiced") == pytest.approx(2.0)
    assert _field(features, "f0_cv") == pytest.approx(
        np.std([100, 150, 200, 250]) / np.mean([100, 150, 200, 250])
    )
    assert captured["sr"] == sr
    assert captured["hop_length"] == hop_length
    assert captured["fmin"] == 80
    assert captured["fmax"] == 400


def test_extract_audio_features_handles_silence_without_nan_or_infinity():
    sr = 1_000
    audio = np.zeros(sr * 2, dtype=np.float32)

    def no_pitch(*args, **kwargs):
        return (
            np.full(5, np.nan),
            np.zeros(5, dtype=bool),
            np.zeros(5, dtype=float),
        )

    features = bd.extract_audio_features(audio, sr, pyin_fn=no_pitch)

    assert _field(features, "voiced_ratio") == 0.0
    assert _field(features, "longest_voiced") == 0.0
    assert _field(features, "f0_cv") is None
    assert np.isfinite(_field(features, "rms_db"))
    json.dumps(_json_ready(features), allow_nan=False)


def test_extract_audio_features_stable_tone_has_more_continuous_voicing_than_paused_fm():
    pytest.importorskip("librosa")
    sr = 16_000
    duration = 4.0
    t = np.arange(int(sr * duration)) / sr
    stable = 0.2 * np.sin(2 * np.pi * 220 * t)

    paused_fm = np.zeros_like(stable)
    voiced_samples = int(0.45 * sr)
    period_samples = int(0.75 * sr)
    for start in range(0, len(paused_fm) - voiced_samples + 1, period_samples):
        local_t = np.arange(voiced_samples) / sr
        frequency = 190 + 100 * np.sin(2 * np.pi * 3 * local_t)
        phase = 2 * np.pi * np.cumsum(frequency) / sr
        paused_fm[start : start + voiced_samples] = 0.2 * np.sin(phase)

    stable_features = bd.extract_audio_features(stable, sr, hop_length=256)
    speech_features = bd.extract_audio_features(paused_fm, sr, hop_length=256)

    assert _field(stable_features, "voiced_ratio") > _field(
        speech_features, "voiced_ratio"
    )
    assert _field(stable_features, "longest_voiced") > _field(
        speech_features, "longest_voiced"
    )


# ---------------------------------------------------------------------------
# Incremental filter cache


def test_filter_slices_only_analyzes_new_slices_and_preserves_existing_results(
    pipeline_paths,
):
    old_slice = _write_large_audio(pipeline_paths.slices / "old.wav")
    new_slice = _write_large_audio(pipeline_paths.slices / "new.wav")
    old_result = {
        "path": str(old_slice),
        "keep": True,
        "reason": None,
        **_feature_values(voiced_ratio=0.42),
    }
    pipeline_paths.filters.write_text(
        json.dumps([old_result], ensure_ascii=False), encoding="utf-8"
    )
    analyzed = []

    def analyzer(path, *args, **kwargs):
        analyzed.append(Path(path))
        return {"keep": True, "reason": None, **_feature_values()}

    results = bd.filter_slices(
        bd.FilterThresholds(), force=False, analyzer=analyzer
    )

    assert analyzed == [new_slice]
    by_name = {Path(item["path"]).name: item for item in results}
    assert set(by_name) == {"old.wav", "new.wav"}
    assert by_name["old.wav"]["voiced_ratio"] == 0.42
    assert old_slice.exists() and new_slice.exists()
    assert len(json.loads(pipeline_paths.filters.read_text(encoding="utf-8"))) == 2


def test_filter_slices_force_reanalyzes_cached_slices(pipeline_paths):
    audio = _write_large_audio(pipeline_paths.slices / "cached.wav")
    pipeline_paths.filters.write_text(
        json.dumps(
            [
                {
                    "path": str(audio),
                    "keep": True,
                    "reason": None,
                    **_feature_values(),
                }
            ]
        ),
        encoding="utf-8",
    )
    analyzed = []

    def analyzer(path, *args, **kwargs):
        analyzed.append(Path(path))
        return {
            "keep": False,
            "reason": "singing",
            **_feature_values(voiced_ratio=0.9, f0_cv=0.05),
        }

    results = bd.filter_slices(bd.FilterThresholds(), force=True, analyzer=analyzer)

    assert analyzed == [audio]
    assert len(results) == 1
    assert results[0]["keep"] is False
    assert results[0]["reason"] == "singing"


def test_filter_slices_ignores_derivatives_from_unlisted_sources(pipeline_paths):
    whitelisted_source = pipeline_paths.wav / "保留.wav"
    unlisted_source = pipeline_paths.wav / "排除.wav"
    whitelisted_source.touch()
    unlisted_source.touch()
    pipeline_paths.assets.write_text(
        f"{whitelisted_source.name}\n", encoding="utf-8"
    )
    kept_slice = _write_large_audio(
        pipeline_paths.slices / "保留_(Vocals)_model__0000000000_0000160000.wav"
    )
    unlisted_slice = _write_large_audio(
        pipeline_paths.slices / "排除_(Vocals)_model__0000000000_0000160000.wav"
    )
    analyzed = []

    def analyzer(path, *args, **kwargs):
        analyzed.append(Path(path))
        return {"keep": True, "reason": None, **_feature_values()}

    results = bd.filter_slices(bd.FilterThresholds(), analyzer=analyzer)

    assert analyzed == [kept_slice]
    assert [Path(item["path"]) for item in results] == [kept_slice]
    assert unlisted_slice.exists()


# ---------------------------------------------------------------------------
# Filter-aware incremental ASR


def test_transcribe_only_processes_kept_uncached_slices_and_merges_results(
    pipeline_paths,
):
    cached = _write_large_audio(pipeline_paths.slices / "cached.wav")
    rejected = _write_large_audio(pipeline_paths.slices / "rejected.wav")
    pending = _write_large_audio(pipeline_paths.slices / "pending.wav")
    pipeline_paths.filters.write_text(
        json.dumps(
            [
                {"path": str(cached), "keep": True, "reason": None},
                {"path": str(rejected), "keep": False, "reason": "singing"},
                {
                    "path": "data/slices/pending.wav",
                    "keep": True,
                    "reason": None,
                },
            ]
        ),
        encoding="utf-8",
    )
    existing = {
        "path": "data/slices/cached.wav",
        "lang": "zh",
        "text": "已有缓存",
    }
    pipeline_paths.asr.write_text(
        json.dumps([existing], ensure_ascii=False), encoding="utf-8"
    )
    transcribed = []

    class FakeModel:
        def transcribe(self, path, **kwargs):
            transcribed.append(Path(path))
            return [SimpleNamespace(text=" 新しい結果 ")], SimpleNamespace(language="ja")

    loader = Mock(return_value=FakeModel())

    results = bd.transcribe("fake-model", model_loader=loader)

    loader.assert_called_once()
    assert transcribed == [pending]
    by_name = {Path(item["path"]).name: item for item in results}
    assert set(by_name) == {"cached.wav", "pending.wav"}
    assert by_name["cached.wav"]["text"] == "已有缓存"
    assert by_name["pending.wav"]["lang"] == "ja"
    assert by_name["pending.wav"]["text"] == "新しい結果"
    assert _resolved_from_project(
        by_name["pending.wav"]["path"], pipeline_paths.root
    ) == pending.resolve()
    assert rejected.name not in {Path(item["path"]).name for item in results}
    assert json.loads(pipeline_paths.asr.read_text(encoding="utf-8")) == results


def test_transcribe_does_not_load_model_when_relative_cache_matches_absolute_slice(
    pipeline_paths,
):
    audio = _write_large_audio(pipeline_paths.slices / "done.wav")
    pipeline_paths.filters.write_text(
        json.dumps([{"path": str(audio), "keep": True, "reason": None}]),
        encoding="utf-8",
    )
    existing = [
        {"path": "data/slices/done.wav", "lang": "ja", "text": "完了"}
    ]
    pipeline_paths.asr.write_text(
        json.dumps(existing, ensure_ascii=False), encoding="utf-8"
    )
    loader = Mock(side_effect=AssertionError("Whisper model must stay lazy"))

    results = bd.transcribe("fake-model", model_loader=loader)

    assert len(results) == 1
    assert results[0]["lang"] == "ja"
    assert results[0]["text"] == "完了"
    assert _resolved_from_project(results[0]["path"], pipeline_paths.root) == audio.resolve()
    loader.assert_not_called()


def test_transcribe_does_not_load_model_when_all_slices_are_filtered_out(
    pipeline_paths,
):
    audio = _write_large_audio(pipeline_paths.slices / "singing.wav")
    pipeline_paths.filters.write_text(
        json.dumps([{"path": str(audio), "keep": False, "reason": "singing"}]),
        encoding="utf-8",
    )
    loader = Mock(side_effect=AssertionError("Whisper model must stay lazy"))

    assert bd.transcribe("fake-model", model_loader=loader) == []
    loader.assert_not_called()


def test_select_asr_slices_accepts_uncertain_but_can_run_accept_only(pipeline_paths):
    paths = [
        _write_large_audio(pipeline_paths.slices / f"{name}.wav")
        for name in ("accept", "uncertain", "reject")
    ]
    filters = [
        {"path": str(path), "keep": True, "reason": None} for path in paths
    ]
    speakers = [
        {"path": str(paths[0]), "decision": "accept"},
        {"path": str(paths[1]), "decision": "uncertain"},
        {"path": str(paths[2]), "decision": "reject"},
    ]

    assert bd.select_asr_slices(
        paths, filters, speaker_results=speakers
    ) == paths[:2]
    assert bd.select_asr_slices(
        paths, filters, speaker_results=speakers, speaker_policy="accept"
    ) == paths[:1]


# ---------------------------------------------------------------------------
# Dataset text/language/filter quality gates


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("啊啊啊啊啊", True),
        ("哈哈哈你好", True),
        ("哈 哈 呀", True),
        ("哈哈你好", False),
        ("正常文本", False),
        ("", False),
        ("   ", False),
    ],
)
def test_is_repetitive_text_ignores_whitespace_and_uses_strict_majority(
    text, expected
):
    assert bd.is_repetitive_text(text) is expected


def test_build_dataset_keeps_only_supported_languages_and_maps_tags(pipeline_paths):
    samples = [
        ("ja", "こんにちは今日も配信だよ"),
        ("zh", "大家好今天也来直播啦"),
        ("en", "hello everyone nice to meet you"),
        ("nn", "nn normal text"),
        ("vi", "vi normal text"),
        ("es", "es normal text"),
        ("ko", "ko normal text"),
        ("yue", "yue normal text"),
    ]
    transcripts = []
    filters = []
    for lang, text in samples:
        source = _write_large_audio(pipeline_paths.slices / f"{lang}.wav")
        transcripts.append({"path": str(source), "lang": lang, "text": text})
        filters.append({"path": str(source), "keep": True, "reason": None})

    list_path = bd.build_dataset(
        transcripts=transcripts, filter_results=filters, allow_en=True
    )
    lines = list_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert lines[0].split("|")[2:] == ["JP", "こんにちは今日も配信だよ"]
    assert lines[1].split("|")[2:] == ["ZH", "大家好今天也来直播啦"]
    assert lines[2].split("|")[2:] == ["EN", "hello everyone nice to meet you"]
    assert sorted(path.name for path in (pipeline_paths.dataset / "audio").glob("*.wav")) == [
        "001_jp.wav",
        "002_zh.wav",
        "003_en.wav",
    ]


def test_build_dataset_drops_en_when_disabled(pipeline_paths):
    source = _write_large_audio(pipeline_paths.slices / "en.wav")
    list_path = bd.build_dataset(
        transcripts=[{"path": str(source), "lang": "en", "text": "hello there"}],
        filter_results=[{"path": str(source), "keep": True, "reason": None}],
        allow_en=False,
    )
    assert list_path.read_text(encoding="utf-8") == ""


def test_build_dataset_applies_filter_annotation_and_other_quality_gates(
    pipeline_paths,
):
    good = _write_large_audio(pipeline_paths.slices / "good.wav")
    filtered = _write_large_audio(pipeline_paths.slices / "filtered.wav")
    short_text = _write_large_audio(pipeline_paths.slices / "short_text.wav")
    tiny = _write_large_audio(pipeline_paths.slices / "tiny.wav", size=100)
    repetitive = _write_large_audio(pipeline_paths.slices / "repetitive.wav")
    unannotated = _write_large_audio(pipeline_paths.slices / "unannotated.wav")
    missing = pipeline_paths.slices / "missing.wav"

    transcripts = [
        {"path": "data/slices/good.wav", "lang": "ja", "text": "正常|文本"},
        {"path": str(filtered), "lang": "zh", "text": "应该过滤"},
        {"path": str(short_text), "lang": "en", "text": "x"},
        {"path": str(tiny), "lang": "ja", "text": "尺寸太小"},
        {"path": str(repetitive), "lang": "zh", "text": "啊啊啊啊呀"},
        {"path": str(unannotated), "lang": "ja", "text": "没有过滤标注"},
        {"path": str(missing), "lang": "ja", "text": "文件不存在"},
    ]
    filters = [
        {"path": str(good), "keep": True, "reason": None},
        {"path": str(filtered), "keep": False, "reason": "singing"},
        {"path": str(short_text), "keep": True, "reason": None},
        {"path": str(tiny), "keep": True, "reason": None},
        {"path": str(repetitive), "keep": True, "reason": None},
        {"path": str(missing), "keep": True, "reason": None},
    ]

    list_path = bd.build_dataset(transcripts=transcripts, filter_results=filters)

    assert list_path.read_text(encoding="utf-8").splitlines() == [
        "audio/001_jp.wav|花音|JP|正常/文本"
    ]
    assert [path.name for path in (pipeline_paths.dataset / "audio").glob("*.wav")] == [
        "001_jp.wav"
    ]
