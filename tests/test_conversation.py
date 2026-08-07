import pytest

from dialogue.conversation import Conversation


def add_turn(conversation: Conversation, index: int) -> None:
    conversation.add_user_message(f"msg{index}")
    conversation.add_assistant_message(f"reply{index}")


def test_add_and_get_messages_returns_a_copy():
    conversation = Conversation()
    add_turn(conversation, 0)

    messages = conversation.get_messages()
    messages[0]["content"] = "changed"

    assert conversation.get_messages() == [
        {"role": "user", "content": "msg0"},
        {"role": "assistant", "content": "reply0"},
    ]


def test_compaction_preserves_recent_turns_and_pending_user():
    conversation = Conversation(
        recent_turns=2,
        summary_trigger_turns=3,
        summary_trigger_chars=10_000,
    )
    for index in range(4):
        add_turn(conversation, index)
    conversation.add_user_message("pending")

    plan = conversation.plan_compaction()

    assert plan is not None
    assert [message["content"] for message in plan.messages()] == [
        "msg0",
        "reply0",
        "msg1",
        "reply1",
    ]
    assert conversation.apply_compaction(plan, "用户此前讨论了前两轮。") is True
    assert [message["content"] for message in conversation.get_messages()] == [
        "msg2",
        "reply2",
        "msg3",
        "reply3",
        "pending",
    ]


def test_context_places_summary_before_recent_messages():
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=2,
        summary_trigger_chars=10_000,
    )
    add_turn(conversation, 0)
    add_turn(conversation, 1)
    plan = conversation.plan_compaction()
    assert plan is not None
    conversation.apply_compaction(plan, "用户叫小明，喜欢爵士乐。")

    context = conversation.get_context_messages()

    assert context[0]["role"] == "system"
    assert "用户叫小明" in context[0]["content"]
    assert [message["content"] for message in context[1:]] == [
        "msg1",
        "reply1",
    ]


def test_compaction_can_trigger_on_character_budget():
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=99,
        summary_trigger_chars=10,
    )
    add_turn(conversation, 0)
    add_turn(conversation, 1)

    assert conversation.plan_compaction() is not None


def test_stale_compaction_plan_cannot_delete_changed_history():
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=2,
        summary_trigger_chars=10_000,
    )
    add_turn(conversation, 0)
    add_turn(conversation, 1)
    plan = conversation.plan_compaction()
    assert plan is not None
    conversation.clear()
    add_turn(conversation, 9)

    assert conversation.apply_compaction(plan, "stale") is False
    assert conversation.get_messages()[0]["content"] == "msg9"


def test_rollback_only_removes_pending_user_and_keeps_summary():
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=2,
        summary_trigger_chars=10_000,
    )
    add_turn(conversation, 0)
    add_turn(conversation, 1)
    plan = conversation.plan_compaction()
    assert plan is not None
    conversation.apply_compaction(plan, "earlier memory")
    conversation.add_user_message("pending")

    conversation.rollback_last_user_message()

    assert conversation.summary == "earlier memory"
    assert [message["content"] for message in conversation.get_messages()] == [
        "msg1",
        "reply1",
    ]


def test_clear_removes_messages_and_summary():
    conversation = Conversation(
        recent_turns=1,
        summary_trigger_turns=2,
        summary_trigger_chars=10_000,
    )
    add_turn(conversation, 0)
    add_turn(conversation, 1)
    plan = conversation.plan_compaction()
    assert plan is not None
    conversation.apply_compaction(plan, "memory")

    conversation.clear()

    assert conversation.get_messages() == []
    assert conversation.summary == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recent_turns": 0},
        {"recent_turns": 2, "summary_trigger_turns": 2},
        {"summary_trigger_chars": 0},
        {"summary_max_chars": 0},
        {"max_turns": 3, "recent_turns": 2},
    ],
)
def test_invalid_memory_settings(kwargs):
    with pytest.raises(ValueError):
        Conversation(**kwargs)
