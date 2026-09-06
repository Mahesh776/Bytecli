from collections.abc import Sequence

from bytecli.memory.base import MemoryStore
from bytecli.memory.types import MemoryEntry, MemoryLevel, MemoryStats, estimate_tokens


class SessionMemory(MemoryStore):
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._entries: dict[str, MemoryEntry] = {}
        self._compaction_count = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    async def store(self, entry: MemoryEntry) -> None:
        entry.level = MemoryLevel.SESSION
        self._entries[entry.key] = entry

    async def retrieve(self, key: str, level: MemoryLevel | None = None) -> MemoryEntry | None:
        return self._entries.get(key)

    async def delete(self, key: str, level: MemoryLevel | None = None) -> None:
        self._entries.pop(key, None)

    async def clear(self, level: MemoryLevel | None = None) -> None:
        self._entries.clear()

    async def stats(self, level: MemoryLevel | None = None) -> MemoryStats:
        return _compute_stats(list(self._entries.values()), self._compaction_count)

    async def search(self, query: str, level: MemoryLevel | None = None, limit: int = 10) -> list[MemoryEntry]:
        q = query.lower()
        matches = [e for e in self._entries.values() if q in e.content.lower() or q in e.key.lower()]
        matches.sort(key=lambda e: e.importance, reverse=True)
        return matches[:limit]

    async def get_all(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    async def compact(self, max_entries: int = 100) -> int:
        if len(self._entries) <= max_entries:
            return 0

        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.timestamp),
        )
        to_remove = sorted_entries[: len(self._entries) - max_entries]
        for entry in to_remove:
            del self._entries[entry.key]

        self._compaction_count += 1
        return len(to_remove)


def _compute_stats(entries: Sequence[MemoryEntry], compaction_count: int) -> MemoryStats:
    if not entries:
        return MemoryStats()

    total_tokens = sum(e.token_count for e in entries) or sum(estimate_tokens(e.content) for e in entries)
    level_counts: dict[MemoryLevel, int] = {}
    for e in entries:
        level_counts[e.level] = level_counts.get(e.level, 0) + 1

    timestamps = [e.timestamp for e in entries]
    return MemoryStats(
        total_entries=len(entries),
        total_tokens=total_tokens,
        level_counts=level_counts,
        oldest_entry=min(timestamps),
        newest_entry=max(timestamps),
        compaction_count=compaction_count,
    )
