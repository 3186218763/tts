from config import load_config


def _base_yaml() -> str:
    return """
llm:
  api_key: key
  base_url: https://example.test
  model: model
tts:
  base_url: http://localhost:9880
  ref_audio_path: /ref.wav
  ref_text: ref
  ref_language: zh
"""


def test_load_config_uses_asr_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(_base_yaml(), encoding="utf-8")

    config = load_config(str(path))

    assert config.asr.model == "small"
    assert config.asr.device == "auto"
    assert config.asr.language == "auto"
    assert config.asr.max_upload_mb == 15


def test_load_config_reads_asr_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        _base_yaml()
        + """
asr:
  model: medium
  device: cuda
  compute_type: float16
  language: zh
  beam_size: 3
  max_upload_mb: 8
""",
        encoding="utf-8",
    )

    config = load_config(str(path))

    assert config.asr.model == "medium"
    assert config.asr.device == "cuda"
    assert config.asr.compute_type == "float16"
    assert config.asr.language == "zh"
    assert config.asr.beam_size == 3
    assert config.asr.max_upload_mb == 8
