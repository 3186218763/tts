from unittest.mock import AsyncMock

import pytest

from dialogue.conversation import Conversation
from dialogue.memory import compact_conversation, prepare_chat_messages


def add_turn(conversation: Conversation, index: int) -> None:
    conversation.add_user_message(f"user-{index}")
    conversation.add_assistant_message(f"assistant-{index}")


@pytest.mark.asyncio
async def test_prepare_messages_compacts_old_turns_and_injects_memory():
    conversation = Conversation(
        recent_turns=2,
        summary_trigger_turns=3,
        summary_trigger_chars=10_000,
    )
    for index in range(4):
        add_turn(conversation, index)
    conversation.add_user_message("current")
    llm = AsyncMock()
    llm.summarize_chat = AsyncMock(return_value="用户早先讨论了0和1。")

    messages = await prepare_chat_messages(llm, conversation)

    llm.summarize_chat.assert_awaited_once()
    assert messages[0]["role"] == "system"
    assert "真白花音" in messages[0]["content"]
    # 注入段（facts/examples）位于人设卡与记忆之间，均为 system 角色
    memory_index = next(
        index
        for index, message in enumerate(messages)
        if "用户早先讨论了0和1" in message["content"]
    )
    assert memory_index >= 2, "旧实现（无注入）下 memory_index==1，此断言保证红阶段有效"
    assert all(message["role"] == "system" for message in messages[1:memory_index])
    assert [message["content"] for message in messages[memory_index + 1 :]] == [
        "user-2",
        "assistant-2",
        "user-3",
        "assistant-3",
        "current",
    ]


@pytest.mark.asyncio
async def test_empty_conversation_skips_persona_injection():
    conversation = Conversation(recent_turns=2, summary_trigger_turns=3)
    llm = AsyncMock()

    messages = await prepare_chat_messages(llm, conversation)

    assert len(messages) == 1
    assert messages[0]["role"] == "system"


@pytest.mark.asyncio
async def test_persona_context_uses_last_user_message():
    conversation = Conversation(recent_turns=2, summary_trigger_turns=3)
    conversation.add_user_message("为什么毕业")
    conversation.add_assistant_message("……")
    llm = AsyncMock()

    messages = await prepare_chat_messages(llm, conversation)

    facts = [m for m in messages if "<persona_facts>" in m["content"]]
    assert facts, "应基于最后一条 user 消息注入事实"
    assert "F012" in facts[0]["content"]


@pytest.mark.asyncio
async def test_summary_failure_keeps_raw_history_and_main_context():
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=2,
        summary_trigger_chars=10_000,
    )
    add_turn(conversation, 0)
    add_turn(conversation, 1)
    llm = AsyncMock()
    llm.summarize_chat = AsyncMock(side_effect=RuntimeError("summary offline"))

    compacted = await compact_conversation(llm, conversation)

    assert compacted is False
    assert len(conversation.get_messages()) == 4
    assert conversation.summary == ""


@pytest.mark.asyncio
async def test_missing_summarizer_degrades_without_losing_history():
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=2,
        summary_trigger_chars=10_000,
    )
    add_turn(conversation, 0)
    add_turn(conversation, 1)

    assert await compact_conversation(object(), conversation) is False
    assert len(conversation.get_messages()) == 4


@pytest.mark.asyncio
async def test_fifty_turns_keep_early_facts_in_summary_and_recent_raw_turns():
    conversation = Conversation(
        recent_turns=8,
        summary_trigger_turns=12,
        summary_trigger_chars=100_000,
        summary_max_chars=10_000,
    )

    class DeterministicSummarizer:
        async def summarize_chat(
            self, *, previous_summary, messages, max_chars
        ):
            archived_users = [
                message["content"]
                for message in messages
                if message["role"] == "user"
            ]
            return " | ".join(
                value for value in [previous_summary, *archived_users] if value
            )[:max_chars]

    llm = DeterministicSummarizer()
    for index in range(50):
        user_text = "我叫小明，喜欢爵士乐" if index == 0 else f"第{index}轮"
        conversation.add_user_message(user_text)
        await compact_conversation(llm, conversation)
        conversation.add_assistant_message(f"回复{index}")

    # Prepare one more in-flight turn after several rolling compactions.
    conversation.add_user_message("还记得我吗")
    await compact_conversation(llm, conversation)

    assert "我叫小明，喜欢爵士乐" in conversation.summary
    raw_messages = conversation.get_messages()
    # Compaction has hysteresis: the raw window grows from recent_turns up to
    # summary_trigger_turns - 1 before the next summary call.
    assert len(raw_messages) <= (12 - 1) * 2 + 1
    assert [message["content"] for message in raw_messages[-17:-1:2]] == [
        f"第{index}轮" for index in range(42, 50)
    ]
    assert raw_messages[-1]["content"] == "还记得我吗"
