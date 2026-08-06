from pathlib import Path

from scripts.run_huayin_api import audio_compatibility_source, runtime_config


def test_audio_compatibility_source_patches_torchaudio_without_torchcodec():
    source = audio_compatibility_source()
    assert "soundfile" in source
    assert "torchaudio.load = _soundfile_load" in source


def test_runtime_config_points_to_trained_v2pro_weights(tmp_path):
    root = tmp_path / "GPT-SoVITS"
    payload = runtime_config(
        root,
        tmp_path / "huayin.ckpt",
        tmp_path / "huayin.pth",
        device="cuda:0",
    )
    custom = payload["custom"]
    assert custom["version"] == "v2Pro"
    assert custom["device"] == "cuda:0"
    assert custom["is_half"] is True
    assert custom["t2s_weights_path"] == str((tmp_path / "huayin.ckpt").resolve())
    assert custom["vits_weights_path"] == str((tmp_path / "huayin.pth").resolve())
    assert custom["cnhuhbert_base_path"].endswith("GPT_SoVITS/pretrained_models/chinese-hubert-base")


def test_runtime_config_disables_half_precision_when_requested(tmp_path):
    payload = runtime_config(
        tmp_path,
        Path("gpt.ckpt"),
        Path("sovits.pth"),
        device="cuda",
        full_precision=True,
    )
    assert payload["custom"]["is_half"] is False
