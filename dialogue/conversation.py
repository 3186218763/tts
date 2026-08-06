"""对话历史管理，自动截断超出上限的旧消息。"""


class Conversation:
    """管理多轮对话历史。每轮 = user + assistant = 2 条消息。"""

    def __init__(self, max_turns: int = 10):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._messages: list[dict] = []
        self._max_turns = max_turns

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._truncate()

    def add_assistant_message(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._truncate()

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def rollback_last_user_message(self) -> None:
        """Remove a user turn that failed before an assistant reply was saved."""
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()

    def _truncate(self) -> None:
        max_messages = self._max_turns * 2
        # Keep one in-flight user message in addition to complete historical
        # turns. Once its assistant reply arrives, trim the oldest full pair.
        pending_user = bool(
            self._messages and self._messages[-1]["role"] == "user"
        )
        keep_messages = max_messages + (1 if pending_user else 0)
        if len(self._messages) > keep_messages:
            self._messages = self._messages[-keep_messages:]
