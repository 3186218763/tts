from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    protocol: str = "openai"
    temperature: float = 0.8
    max_tokens: int = 400
    frequency_penalty: float = 0.15

    def __post_init__(self):
        if self.protocol not in ("openai", "anthropic"):
            raise ValueError(
                f"llm.protocol 必须是 openai 或 anthropic，当前为 {self.protocol!r}"
            )


@dataclass
class TTSConfig:
    base_url: str
    ref_audio_path: str
    ref_text: str
    ref_language: str
    text_language: str = "auto"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 0.85
    repetition_penalty: float = 1.35
    speed_factor: float = 1.0
    seed: int = 42
    text_split_method: str = "cut5"


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
    max_turns: int = 8
    summary_trigger_turns: int = 12
    summary_trigger_chars: int = 12_000
    summary_max_chars: int = 1_800
    min_sentence_chars: int = 4
    max_sentence_chars: int = 50
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
    conversation = data.get("conversation", {})
    streaming = data.get("streaming", {})
    return AppConfig(
        llm=LLMConfig(**data["llm"]),
        tts=TTSConfig(**data["tts"]),
        asr=ASRConfig(**data.get("asr", {})),
        max_turns=conversation.get(
            "recent_turns", conversation.get("max_turns", 8)
        ),
        summary_trigger_turns=conversation.get("summary_trigger_turns", 12),
        summary_trigger_chars=conversation.get("summary_trigger_chars", 12_000),
        summary_max_chars=conversation.get("summary_max_chars", 1_800),
        min_sentence_chars=streaming.get("min_sentence_chars", 4),
        max_sentence_chars=streaming.get("max_sentence_chars", 50),
    )
