from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str


@dataclass
class TTSConfig:
    base_url: str
    ref_audio_path: str
    ref_text: str
    ref_language: str


@dataclass
class ASRConfig:
    model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "auto"
    beam_size: int = 5
    max_upload_mb: int = 15


@dataclass
class AppConfig:
    llm: LLMConfig
    tts: TTSConfig
    max_turns: int = 10
    max_sentence_chars: int = 25
    asr: ASRConfig = field(default_factory=ASRConfig)


def load_config(path: str = "configs/config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{path}\n"
            f"请复制 configs/config.example.yaml 为 configs/config.yaml 并填写配置"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(
        llm=LLMConfig(**data["llm"]),
        tts=TTSConfig(**data["tts"]),
        asr=ASRConfig(**data.get("asr", {})),
        max_turns=data.get("conversation", {}).get("max_turns", 10),
        max_sentence_chars=data.get("streaming", {}).get("max_sentence_chars", 25),
    )
