from pathlib import Path
from typing import Any

from bytecli.memory.base import MemoryStore
from bytecli.memory.conversation import ConversationMemory
from bytecli.memory.project import ProjectMemory
from bytecli.memory.session import SessionMemory
from bytecli.memory.types import (
    ConversationStats,
    MemoryEntry,
    MemoryLevel,
    MemoryStats,
    MessageEntry,
)
from bytecli.memory.workspace import WorkspaceMemory


class MemoryManager:
    def __init__(
        self,
        session_id: str,
        workspace_dir: Path | None = None,
        max_context_tokens: int = 32000,
        compaction_threshold: float = 0.8,
        event_bus: object | None = None,
    ) -> None:
        self._conversation = ConversationMemory(
            max_tokens=max_context_tokens,
            compaction_threshold=compaction_threshold,
        )
        self._session = SessionMemory(session_id=session_id)
        self._project: ProjectMemory | None = None
        self._workspace: WorkspaceMemory | None = None
        self._event_bus = event_bus

        if workspace_dir is not None:
            self._project = ProjectMemory(workspace_dir)
            self._workspace = WorkspaceMemory(workspace_dir)

    @property
    def conversation(self) -> ConversationMemory:
        return self._conversation

    @property
    def session(self) -> SessionMemory:
        return self._session

    @property
    def project(self) -> ProjectMemory | None:
        return self._project

    @property
    def workspace(self) -> WorkspaceMemory | None:
        return self._workspace

    async def add_message(self, message: MessageEntry) -> None:
        await self._conversation.add_message(message)

    async def get_context(self, limit: int | None = None) -> list[MessageEntry]:
        return await self._conversation.get_messages(limit=limit)

    async def store_entry(
        self, key: str, content: str, level: MemoryLevel = MemoryLevel.SESSION, **kwargs: Any
    ) -> None:
        entry = MemoryEntry(key=key, content=content, level=level, **kwargs)
        store = self._get_store(level)
        await store.store(entry)

    async def retrieve_entry(self, key: str, level: MemoryLevel | None = None) -> MemoryEntry | None:
        if level is not None:
            store = self._get_store(level)
            return await store.retrieve(key)
        for store in self._all_stores():
            result = await store.retrieve(key)
            if result is not None:
                return result
        return None

    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        results: list[MemoryEntry] = []
        for store in self._all_stores():
            results.extend(await store.search(query, limit=limit))
        results.sort(key=lambda e: e.importance, reverse=True)
        return results[:limit]

    async def conversation_stats(self) -> ConversationStats:
        return await self._conversation.stats()

    async def memory_stats(self) -> dict[str, MemoryStats]:
        stats: dict[str, MemoryStats] = {}
        for name, store in self._named_stores():
            stats[name] = await store.stats()
        return stats

    async def compact(self) -> dict[str, int]:
        result: dict[str, int] = {}
        result["conversation"] = await self._conversation.compact()
        result["session"] = await self._session.compact()
        if self._project is not None:
            result["project"] = await self._project.compact()
        if self._workspace is not None:
            result["workspace"] = await self._workspace.compact()

        if self._event_bus is not None:
            total = sum(result.values())
            if total > 0:
                await self._publish_compaction_event(total)

        return result

    async def clear_all(self) -> None:
        await self._conversation.clear()
        await self._session.clear()
        if self._project is not None:
            await self._project.clear()
        if self._workspace is not None:
            await self._workspace.clear()

    def _get_store(self, level: MemoryLevel) -> MemoryStore:
        if level == MemoryLevel.CONVERSATION:
            raise ValueError("Use add_message/get_context for conversation memory")
        if level == MemoryLevel.SESSION:
            return self._session
        if level == MemoryLevel.PROJECT:
            if self._project is None:
                raise ValueError("Project memory not initialized (no workspace_dir)")
            return self._project
        if level == MemoryLevel.WORKSPACE:
            if self._workspace is None:
                raise ValueError("Workspace memory not initialized (no workspace_dir)")
            return self._workspace
        raise ValueError(f"Unknown memory level: {level}")

    def _all_stores(self) -> list[MemoryStore]:
        stores: list[MemoryStore] = [self._session]
        if self._project is not None:
            stores.append(self._project)
        if self._workspace is not None:
            stores.append(self._workspace)
        return stores

    def _named_stores(self) -> list[tuple[str, MemoryStore]]:
        stores: list[tuple[str, MemoryStore]] = [
            ("session", self._session),
        ]
        if self._project is not None:
            stores.append(("project", self._project))
        if self._workspace is not None:
            stores.append(("workspace", self._workspace))
        return stores

    async def _publish_compaction_event(self, total_removed: int) -> None:
        if hasattr(self._event_bus, "publish"):
            from bytecli.core.events import Event

            event = Event(
                type="memory.compaction.end",
                timestamp=__import__("time").time(),
                data={"total_removed": total_removed},
            )
            await self._event_bus.publish(event)
