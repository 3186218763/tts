"""Shared rolling-memory preparation for CLI and Web conversations."""

from __future__ import annotations

from .conversation import Conversation
from .persona import get_system_prompt


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


async def prepare_chat_messages(
    llm_client, conversation: Conversation
) -> list[dict[str, str]]:
    """Compact if needed, then build the ordered model context."""
    await compact_conversation(llm_client, conversation)
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(conversation.get_context_messages())
    return messages
