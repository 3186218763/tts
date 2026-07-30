from dialogue.conversation import Conversation


def test_add_and_get_messages():
    conv = Conversation(max_turns=10)
    conv.add_user_message("你好")
    conv.add_assistant_message("你好呀")
    messages = conv.get_messages()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "你好"}
    assert messages[1] == {"role": "assistant", "content": "你好呀"}


def test_max_turns_truncation():
    conv = Conversation(max_turns=2)
    for i in range(3):
        conv.add_user_message(f"msg{i}")
        conv.add_assistant_message(f"reply{i}")
    messages = conv.get_messages()
    # max_turns=2 → 保留最近 2 轮 = 4 条消息
    assert len(messages) == 4
    assert messages[0]["content"] == "msg1"
    assert messages[3]["content"] == "reply2"


def test_clear():
    conv = Conversation()
    conv.add_user_message("你好")
    conv.clear()
    assert conv.get_messages() == []


def test_empty_initial():
    assert Conversation().get_messages() == []
