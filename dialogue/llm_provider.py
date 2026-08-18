"""Shared provider defaults and compatibility handling for LLM transports."""

from typing import Final


SUPPORTED_LLM_PROVIDERS: Final = ("openai", "anthropic", "gemini")
DEFAULT_LLM_BASE_URLS: Final = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}


def resolve_llm_provider(
    *, provider: str | None = None, protocol: str | None = None
) -> str:
    """Return one normalized provider while accepting the legacy protocol key."""
    normalized_provider = _normalize(provider, "provider")
    normalized_protocol = _normalize(protocol, "protocol")
    if (
        normalized_provider
        and normalized_protocol
        and normalized_provider != normalized_protocol
    ):
        raise ValueError("provider 与 protocol 不能同时指定不同的值")
    return normalized_provider or normalized_protocol or "openai"


def _normalize(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串")
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in SUPPORTED_LLM_PROVIDERS:
        choices = "、".join(SUPPORTED_LLM_PROVIDERS)
        raise ValueError(
            f"{field_name} 必须是 {choices}，当前为 {value!r}"
        )
    return normalized
