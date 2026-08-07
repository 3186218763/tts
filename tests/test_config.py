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
    assert config.llm.temperature == 0.8
    assert config.max_turns == 8
    assert config.summary_trigger_turns == 12
    assert config.min_sentence_chars == 4
    assert config.max_sentence_chars == 50


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


def test_load_config_reads_long_conversation_and_llm_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        _base_yaml()
        + """
conversation:
  recent_turns: 6
  summary_trigger_turns: 10
  summary_trigger_chars: 9000
  summary_max_chars: 1200
streaming:
  min_sentence_chars: 3
  max_sentence_chars: 60
""",
        encoding="utf-8",
    )

    config = load_config(str(path))

    assert config.max_turns == 6
    assert config.summary_trigger_turns == 10
    assert config.summary_trigger_chars == 9000
    assert config.summary_max_chars == 1200
    assert config.min_sentence_chars == 3
    assert config.max_sentence_chars == 60


def test_load_config_supports_legacy_max_turns(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        _base_yaml() + "\nconversation:\n  max_turns: 7\n",
        encoding="utf-8",
    )

    assert load_config(str(path)).max_turns == 7
