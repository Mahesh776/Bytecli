from collections.abc import Sequence

from bytecli.memory.types import (
    ConversationStats,
    MessageEntry,
    estimate_tokens,
    messages_token_count,
)


class ConversationMemory:
    def __init__(
        self,
        max_tokens: int = 32000,
        compaction_threshold: float = 0.8,
    ) -> None:
        self._messages: list[MessageEntry] = []
        self._max_tokens = max_tokens
        self._compaction_threshold = compaction_threshold
        self._compaction_count = 0

    async def add_message(self, message: MessageEntry) -> None:
        if message.token_count == 0 and message.content:
            message.token_count = estimate_tokens(message.content)
        self._messages.append(message)
        if self._needs_compaction():
            await self.compact()

    async def add_messages(self, messages: Sequence[MessageEntry]) -> None:
        for msg in messages:
            await self.add_message(msg)

    async def get_messages(self, limit: int | None = None) -> list[MessageEntry]:
        if limit is not None and limit < len(self._messages):
            return self._messages[-limit:]
        return list(self._messages)

    async def get_token_count(self) -> int:
        return messages_token_count(self._messages)

    async def compact(self) -> int:
        if not self._messages:
            return 0

        current_tokens = await self.get_token_count()
        target_tokens = int(self._max_tokens * self._compaction_threshold)
        if current_tokens <= target_tokens:
            return 0

        removed = 0
        while self._messages and current_tokens > target_tokens:
            oldest = self._messages[0]
            current_tokens -= oldest.token_count
            self._messages.pop(0)
            removed += 1

        self._compaction_count += 1
        return removed

    async def clear(self) -> None:
        self._messages.clear()

    async def stats(self) -> ConversationStats:
        if not self._messages:
            return ConversationStats()

        return ConversationStats(
            message_count=len(self._messages),
            total_tokens=await self.get_token_count(),
            oldest_message=self._messages[0].timestamp,
            newest_message=self._messages[-1].timestamp,
            compaction_count=self._compaction_count,
        )

    async def export_messages(self) -> list[dict[str, object]]:
        return [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
            }
            for m in self._messages
        ]

    def _needs_compaction(self) -> bool:
        return self._max_tokens > 0 and messages_token_count(self._messages) > self._max_tokens
