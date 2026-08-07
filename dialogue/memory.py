"""Shared rolling-memory preparation for CLI and Web conversations."""

from __future__ import annotations

from .conversation import Conversation
from .persona import get_system_prompt
from .persona_context import build_persona_context


async def compact_conversation(llm_client, conversation: Conversation) -> bool:
    """Compact old complete turns when supported, degrading safely on failure."""
    plan = conversation.plan_compaction()
    if plan is None:
        return False
    summarize = getattr(llm_client, "summarize_chat", None)
    if not callable(summarize):
        return False
    try:
        summary = await summarize(
            previous_summary=plan.previous_summary,
            messages=plan.messages(),
            max_chars=conversation.summary_max_chars,
        )
    except Exception:
        # Memory maintenance is auxiliary. The full raw history remains intact
        # and the main response can still proceed when summarization is down.
        return False
    return conversation.apply_compaction(plan, summary)


def _last_user_text(conversation: Conversation) -> str | None:
    """回扫取最后一条 role=user 消息（容忍尾消息为 assistant 的补答/重试场景）；空会话返回 None。"""
    for message in reversed(conversation.get_messages()):
        if message["role"] == "user":
            return message["content"]
    return None


async def prepare_chat_messages(
    llm_client, conversation: Conversation
) -> list[dict[str, str]]:
    """Compact if needed, then build the ordered model context."""
    await compact_conversation(llm_client, conversation)
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(build_persona_context(_last_user_text(conversation)))
    messages.extend(conversation.get_context_messages())
    return messages
