import json
import time
from pathlib import Path

from bytecli.memory.base import MemoryStore
from bytecli.memory.session import _compute_stats
from bytecli.memory.types import MemoryEntry, MemoryLevel, MemoryScope, MemoryStats


class ProjectMemory(MemoryStore):
    def __init__(self, project_dir: Path) -> None:
        self._project_dir = project_dir
        self._entries: dict[str, MemoryEntry] = {}
        self._compaction_count = 0
        self._dirty = False
        self._load()

    def _store_path(self) -> Path:
        return self._project_dir / ".bytecli" / "memory.json"

    def _load(self) -> None:
        path = self._store_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text("utf-8"))
            for item in data.get("entries", []):
                entry = MemoryEntry(
                    key=item["key"],
                    content=item["content"],
                    level=MemoryLevel(item.get("level", "project")),
                    timestamp=item.get("timestamp", time.time()),
                    metadata=item.get("metadata", {}),
                    token_count=item.get("token_count", 0),
                    importance=item.get("importance", 0.0),
                    scope=MemoryScope(item.get("scope", "persistent")),
                )
                self._entries[entry.key] = entry
            self._compaction_count = data.get("compaction_count", 0)
        except (json.JSONDecodeError, KeyError):
            pass

    def _save(self) -> None:
        if not self._dirty:
            return
        path = self._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [
                {
                    "key": e.key,
                    "content": e.content,
                    "level": str(e.level),
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                    "token_count": e.token_count,
                    "importance": e.importance,
                    "scope": str(e.scope),
                }
                for e in self._entries.values()
            ],
            "compaction_count": self._compaction_count,
        }
        path.write_text(json.dumps(data, indent=2), "utf-8")

    async def store(self, entry: MemoryEntry) -> None:
        entry.level = MemoryLevel.PROJECT
        entry.scope = MemoryScope.PERSISTENT
        self._entries[entry.key] = entry
        self._dirty = True
        self._save()

    async def retrieve(self, key: str, level: MemoryLevel | None = None) -> MemoryEntry | None:
        return self._entries.get(key)

    async def delete(self, key: str, level: MemoryLevel | None = None) -> None:
        self._entries.pop(key, None)
        self._dirty = True
        self._save()

    async def clear(self, level: MemoryLevel | None = None) -> None:
        self._entries.clear()
        self._dirty = True
        self._save()

    async def stats(self, level: MemoryLevel | None = None) -> MemoryStats:
        return _compute_stats(list(self._entries.values()), self._compaction_count)

    async def search(self, query: str, level: MemoryLevel | None = None, limit: int = 10) -> list[MemoryEntry]:
        q = query.lower()
        matches = [e for e in self._entries.values() if q in e.content.lower() or q in e.key.lower()]
        matches.sort(key=lambda e: e.importance, reverse=True)
        return matches[:limit]

    async def get_all(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    async def compact(self, max_entries: int = 200) -> int:
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
        self._dirty = True
        self._save()
        return len(to_remove)
