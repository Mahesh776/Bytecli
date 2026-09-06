import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


class MemoryLevel(StrEnum):
    CONVERSATION = auto()
    SESSION = auto()
    PROJECT = auto()
    WORKSPACE = auto()


class MemoryScope(StrEnum):
    EPHEMERAL = auto()
    PERSISTENT = auto()


@dataclass
class MemoryEntry:
    key: str
    content: str
    level: MemoryLevel
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    importance: float = 0.0
    scope: MemoryScope = MemoryScope.EPHEMERAL


@dataclass
class MessageEntry:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class MemoryStats:
    total_entries: int = 0
    total_tokens: int = 0
    level_counts: dict[MemoryLevel, int] = field(default_factory=dict)
    oldest_entry: float | None = None
    newest_entry: float | None = None
    compaction_count: int = 0


@dataclass
class ConversationStats:
    message_count: int = 0
    total_tokens: int = 0
    oldest_message: float | None = None
    newest_message: float | None = None
    compaction_count: int = 0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def messages_token_count(messages: Sequence[MessageEntry]) -> int:
    return sum(m.token_count for m in messages)
