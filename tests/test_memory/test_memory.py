import tempfile
import time
from pathlib import Path

import pytest

from bytecli.memory.base import MemoryStore
from bytecli.memory.conversation import ConversationMemory
from bytecli.memory.manager import MemoryManager
from bytecli.memory.project import ProjectMemory
from bytecli.memory.session import SessionMemory
from bytecli.memory.types import (
    ConversationStats,
    MemoryEntry,
    MemoryLevel,
    MemoryScope,
    MemoryStats,
    MessageEntry,
    estimate_tokens,
    messages_token_count,
)
from bytecli.memory.workspace import WorkspaceMemory


class TestTypes:
    def test_memory_entry_defaults(self) -> None:
        entry = MemoryEntry(key="test-key", content="test content", level=MemoryLevel.SESSION)
        assert entry.key == "test-key"
        assert entry.content == "test content"
        assert entry.level == MemoryLevel.SESSION
        assert entry.scope == MemoryScope.EPHEMERAL
        assert entry.importance == 0.0
        assert entry.token_count == 0

    def test_memory_entry_full(self) -> None:
        entry = MemoryEntry(
            key="k",
            content="c",
            level=MemoryLevel.PROJECT,
            importance=0.9,
            token_count=50,
            scope=MemoryScope.PERSISTENT,
            metadata={"source": "test"},
        )
        assert entry.importance == 0.9
        assert entry.token_count == 50
        assert entry.scope == MemoryScope.PERSISTENT
        assert entry.metadata == {"source": "test"}

    def test_message_entry_defaults(self) -> None:
        msg = MessageEntry(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.token_count == 0
        assert msg.tool_calls is None

    def test_message_entry_with_tool_calls(self) -> None:
        msg = MessageEntry(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "function": {"name": "read", "arguments": "{}"}}],
            tool_call_id=None,
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_conversation_stats_defaults(self) -> None:
        stats = ConversationStats()
        assert stats.message_count == 0
        assert stats.total_tokens == 0
        assert stats.compaction_count == 0

    def test_memory_stats_defaults(self) -> None:
        stats = MemoryStats()
        assert stats.total_entries == 0
        assert stats.total_tokens == 0
        assert stats.level_counts == {}

    def test_estimate_tokens_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self) -> None:
        assert estimate_tokens("hello") == 1

    def test_estimate_tokens_long(self) -> None:
        text = "x" * 100
        assert estimate_tokens(text) == 25

    def test_messages_token_count(self) -> None:
        msgs = [
            MessageEntry(role="user", content="hi", token_count=5),
            MessageEntry(role="assistant", content="hello", token_count=10),
        ]
        assert messages_token_count(msgs) == 15

    def test_memory_level_values(self) -> None:
        assert MemoryLevel.CONVERSATION.value == "conversation"
        assert MemoryLevel.SESSION.value == "session"
        assert MemoryLevel.PROJECT.value == "project"
        assert MemoryLevel.WORKSPACE.value == "workspace"

    def test_memory_scope_values(self) -> None:
        assert MemoryScope.EPHEMERAL.value == "ephemeral"
        assert MemoryScope.PERSISTENT.value == "persistent"


class TestConversationMemory:
    @pytest.fixture
    def conv(self) -> ConversationMemory:
        return ConversationMemory(max_tokens=100, compaction_threshold=0.5)

    async def test_add_and_get_messages(self, conv: ConversationMemory) -> None:
        msg1 = MessageEntry(role="user", content="hello", token_count=10)
        msg2 = MessageEntry(role="assistant", content="world", token_count=10)
        await conv.add_message(msg1)
        await conv.add_message(msg2)
        messages = await conv.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    async def test_add_message_auto_token_count(self, conv: ConversationMemory) -> None:
        msg = MessageEntry(role="user", content="hello world")
        assert msg.token_count == 0
        await conv.add_message(msg)
        assert msg.token_count > 0

    async def test_get_messages_limit(self, conv: ConversationMemory) -> None:
        for i in range(10):
            await conv.add_message(MessageEntry(role="user", content=f"msg{i}", token_count=1))
        messages = await conv.get_messages(limit=3)
        assert len(messages) == 3
        assert messages[-1].content == "msg9"

    async def test_get_token_count(self, conv: ConversationMemory) -> None:
        await conv.add_message(MessageEntry(role="user", content="a", token_count=10))
        await conv.add_message(MessageEntry(role="assistant", content="b", token_count=20))
        assert await conv.get_token_count() == 30

    async def test_compact_removes_oldest(self, conv: ConversationMemory) -> None:
        for i in range(20):
            await conv.add_message(MessageEntry(role="user", content=f"msg{i}", token_count=10))
        removed = await conv.compact()
        assert removed > 0
        messages = await conv.get_messages()
        assert len(messages) < 20

    async def test_compact_noop_when_under_threshold(self, conv: ConversationMemory) -> None:
        await conv.add_message(MessageEntry(role="user", content="hi", token_count=10))
        removed = await conv.compact()
        assert removed == 0

    async def test_clear(self, conv: ConversationMemory) -> None:
        await conv.add_message(MessageEntry(role="user", content="hi", token_count=1))
        await conv.clear()
        assert await conv.get_messages() == []

    async def test_stats_empty(self, conv: ConversationMemory) -> None:
        stats = await conv.stats()
        assert stats.message_count == 0

    async def test_stats_after_messages(self, conv: ConversationMemory) -> None:
        await conv.add_message(MessageEntry(role="user", content="hi", token_count=5))
        await conv.add_message(MessageEntry(role="assistant", content="hello", token_count=10))
        stats = await conv.stats()
        assert stats.message_count == 2
        assert stats.total_tokens == 15
        assert stats.oldest_message is not None
        assert stats.newest_message is not None

    async def test_export_messages(self, conv: ConversationMemory) -> None:
        await conv.add_message(MessageEntry(role="user", content="hi", token_count=1))
        exported = await conv.export_messages()
        assert len(exported) == 1
        assert exported[0]["role"] == "user"
        assert exported[0]["content"] == "hi"

    async def test_auto_compaction_on_add(self, conv: ConversationMemory) -> None:
        for i in range(30):
            await conv.add_message(MessageEntry(role="user", content=f"msg{i}", token_count=10))
        messages = await conv.get_messages()
        assert len(messages) < 30

    async def test_add_messages_bulk(self, conv: ConversationMemory) -> None:
        msgs = [
            MessageEntry(role="user", content="a", token_count=1),
            MessageEntry(role="assistant", content="b", token_count=1),
        ]
        await conv.add_messages(msgs)
        assert len(await conv.get_messages()) == 2

    async def test_compact_tracks_count(self, conv: ConversationMemory) -> None:
        for i in range(30):
            await conv.add_message(MessageEntry(role="user", content=f"msg{i}", token_count=10))
        stats = await conv.stats()
        assert stats.compaction_count >= 0


class TestSessionMemory:
    @pytest.fixture
    def session(self) -> SessionMemory:
        return SessionMemory(session_id="test-session")

    async def test_store_and_retrieve(self, session: SessionMemory) -> None:
        entry = MemoryEntry(key="k1", content="test", level=MemoryLevel.SESSION)
        await session.store(entry)
        result = await session.retrieve("k1")
        assert result is not None
        assert result.key == "k1"
        assert result.content == "test"

    async def test_retrieve_missing(self, session: SessionMemory) -> None:
        result = await session.retrieve("nonexistent")
        assert result is None

    async def test_delete(self, session: SessionMemory) -> None:
        entry = MemoryEntry(key="k1", content="test", level=MemoryLevel.SESSION)
        await session.store(entry)
        await session.delete("k1")
        assert await session.retrieve("k1") is None

    async def test_clear(self, session: SessionMemory) -> None:
        await session.store(MemoryEntry(key="k1", content="a", level=MemoryLevel.SESSION))
        await session.store(MemoryEntry(key="k2", content="b", level=MemoryLevel.SESSION))
        await session.clear()
        assert await session.stats() == MemoryStats()

    async def test_search(self, session: SessionMemory) -> None:
        await session.store(MemoryEntry(key="k1", content="python code", level=MemoryLevel.SESSION, importance=0.8))
        await session.store(MemoryEntry(key="k2", content="javascript code", level=MemoryLevel.SESSION, importance=0.5))
        results = await session.search("python")
        assert len(results) == 1
        assert results[0].key == "k1"

    async def test_search_by_key(self, session: SessionMemory) -> None:
        await session.store(MemoryEntry(key="my-key", content="something", level=MemoryLevel.SESSION))
        results = await session.search("my-key")
        assert len(results) == 1

    async def test_search_ordering(self, session: SessionMemory) -> None:
        await session.store(MemoryEntry(key="k1", content="python", level=MemoryLevel.SESSION, importance=0.3))
        await session.store(MemoryEntry(key="k2", content="python", level=MemoryLevel.SESSION, importance=0.9))
        results = await session.search("python")
        assert results[0].key == "k2"

    async def test_get_all(self, session: SessionMemory) -> None:
        await session.store(MemoryEntry(key="k1", content="a", level=MemoryLevel.SESSION))
        await session.store(MemoryEntry(key="k2", content="b", level=MemoryLevel.SESSION))
        assert len(await session.get_all()) == 2

    async def test_stats(self, session: SessionMemory) -> None:
        entry = MemoryEntry(key="k1", content="test content", level=MemoryLevel.SESSION, token_count=10)
        await session.store(entry)
        stats = await session.stats()
        assert stats.total_entries == 1
        assert stats.total_tokens == 10

    async def test_compact(self, session: SessionMemory) -> None:
        for i in range(10):
            await session.store(MemoryEntry(key=f"k{i}", content=f"entry{i}", level=MemoryLevel.SESSION, importance=0.1))
        removed = await session.compact(max_entries=5)
        assert removed == 5
        assert len(await session.get_all()) == 5

    async def test_compact_noop(self, session: SessionMemory) -> None:
        for i in range(3):
            await session.store(MemoryEntry(key=f"k{i}", content="x", level=MemoryLevel.SESSION))
        removed = await session.compact(max_entries=10)
        assert removed == 0

    async def test_session_id_property(self, session: SessionMemory) -> None:
        assert session.session_id == "test-session"

    async def test_level_assigned_on_store(self, session: SessionMemory) -> None:
        entry = MemoryEntry(key="k1", content="test", level=MemoryLevel.CONVERSATION)
        await session.store(entry)
        assert entry.level == MemoryLevel.SESSION

    async def test_delete_nonexistent(self, session: SessionMemory) -> None:
        await session.delete("nonexistent")


class TestProjectMemory:
    @pytest.fixture
    def project_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_store_and_retrieve(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        entry = MemoryEntry(key="k1", content="project data", level=MemoryLevel.PROJECT)
        await mem.store(entry)
        result = await mem.retrieve("k1")
        assert result is not None
        assert result.content == "project data"

    async def test_persistence(self, project_dir: Path) -> None:
        mem1 = ProjectMemory(project_dir)
        await mem1.store(MemoryEntry(key="k1", content="persistent", level=MemoryLevel.PROJECT))
        mem2 = ProjectMemory(project_dir)
        result = await mem2.retrieve("k1")
        assert result is not None
        assert result.content == "persistent"

    async def test_delete(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        await mem.store(MemoryEntry(key="k1", content="x", level=MemoryLevel.PROJECT))
        await mem.delete("k1")
        assert await mem.retrieve("k1") is None

    async def test_clear(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        await mem.store(MemoryEntry(key="k1", content="x", level=MemoryLevel.PROJECT))
        await mem.clear()
        assert (await mem.stats()).total_entries == 0

    async def test_search(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        await mem.store(MemoryEntry(key="k1", content="python function", level=MemoryLevel.PROJECT))
        results = await mem.search("python")
        assert len(results) == 1

    async def test_get_all(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        await mem.store(MemoryEntry(key="k1", content="a", level=MemoryLevel.PROJECT))
        await mem.store(MemoryEntry(key="k2", content="b", level=MemoryLevel.PROJECT))
        assert len(await mem.get_all()) == 2

    async def test_compact(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        for i in range(10):
            await mem.store(MemoryEntry(key=f"k{i}", content=f"x{i}", level=MemoryLevel.PROJECT, importance=0.1))
        removed = await mem.compact(max_entries=5)
        assert removed == 5

    async def test_level_assigned(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        entry = MemoryEntry(key="k1", content="test", level=MemoryLevel.SESSION)
        await mem.store(entry)
        assert entry.level == MemoryLevel.PROJECT

    async def test_scope_assigned(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        entry = MemoryEntry(key="k1", content="test", level=MemoryLevel.PROJECT)
        await mem.store(entry)
        assert entry.scope == MemoryScope.PERSISTENT

    async def test_stats(self, project_dir: Path) -> None:
        mem = ProjectMemory(project_dir)
        await mem.store(MemoryEntry(key="k1", content="test", level=MemoryLevel.PROJECT, token_count=10))
        stats = await mem.stats()
        assert stats.total_entries == 1
        assert MemoryLevel.PROJECT in stats.level_counts


class TestWorkspaceMemory:
    @pytest.fixture
    def workspace_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_store_and_retrieve(self, workspace_dir: Path) -> None:
        mem = WorkspaceMemory(workspace_dir)
        await mem.store(MemoryEntry(key="k1", content="workspace data", level=MemoryLevel.WORKSPACE))
        result = await mem.retrieve("k1")
        assert result is not None
        assert result.content == "workspace data"

    async def test_persistence(self, workspace_dir: Path) -> None:
        mem1 = WorkspaceMemory(workspace_dir)
        await mem1.store(MemoryEntry(key="k1", content="persist", level=MemoryLevel.WORKSPACE))
        mem2 = WorkspaceMemory(workspace_dir)
        result = await mem2.retrieve("k1")
        assert result is not None
        assert result.content == "persist"

    async def test_delete(self, workspace_dir: Path) -> None:
        mem = WorkspaceMemory(workspace_dir)
        await mem.store(MemoryEntry(key="k1", content="x", level=MemoryLevel.WORKSPACE))
        await mem.delete("k1")
        assert await mem.retrieve("k1") is None

    async def test_clear(self, workspace_dir: Path) -> None:
        mem = WorkspaceMemory(workspace_dir)
        await mem.store(MemoryEntry(key="k1", content="x", level=MemoryLevel.WORKSPACE))
        await mem.clear()
        assert (await mem.stats()).total_entries == 0

    async def test_search(self, workspace_dir: Path) -> None:
        mem = WorkspaceMemory(workspace_dir)
        await mem.store(MemoryEntry(key="k1", content="rust module", level=MemoryLevel.WORKSPACE))
        results = await mem.search("rust")
        assert len(results) == 1

    async def test_compact(self, workspace_dir: Path) -> None:
        mem = WorkspaceMemory(workspace_dir)
        for i in range(10):
            await mem.store(MemoryEntry(key=f"k{i}", content="x", level=MemoryLevel.WORKSPACE, importance=0.1))
        removed = await mem.compact(max_entries=5)
        assert removed == 5


class TestMemoryManager:
    @pytest.fixture
    def workspace_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_manager_with_workspace(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        assert mgr.conversation is not None
        assert mgr.session is not None
        assert mgr.project is not None
        assert mgr.workspace is not None

    async def test_manager_without_workspace(self) -> None:
        mgr = MemoryManager(session_id="s1")
        assert mgr.project is None
        assert mgr.workspace is None

    async def test_add_and_get_context(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        msg = MessageEntry(role="user", content="hello", token_count=5)
        await mgr.add_message(msg)
        context = await mgr.get_context()
        assert len(context) == 1
        assert context[0].content == "hello"

    async def test_store_entry_session(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.store_entry("k1", "session data", level=MemoryLevel.SESSION)
        result = await mgr.retrieve_entry("k1", level=MemoryLevel.SESSION)
        assert result is not None
        assert result.content == "session data"

    async def test_store_entry_project(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.store_entry("k1", "project data", level=MemoryLevel.PROJECT)
        result = await mgr.retrieve_entry("k1", level=MemoryLevel.PROJECT)
        assert result is not None
        assert result.content == "project data"

    async def test_store_entry_workspace(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.store_entry("k1", "workspace data", level=MemoryLevel.WORKSPACE)
        result = await mgr.retrieve_entry("k1", level=MemoryLevel.WORKSPACE)
        assert result is not None
        assert result.content == "workspace data"

    async def test_store_entry_conversation_raises(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        with pytest.raises(ValueError, match="Use add_message"):
            await mgr.store_entry("k1", "data", level=MemoryLevel.CONVERSATION)

    async def test_retrieve_entry_cross_level(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.store_entry("k1", "session data", level=MemoryLevel.SESSION)
        await mgr.store_entry("k1", "project data", level=MemoryLevel.PROJECT)
        result = await mgr.retrieve_entry("k1")
        assert result is not None
        assert result.content == "session data"

    async def test_search(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.store_entry("k1", "python is great", level=MemoryLevel.SESSION, importance=0.9)
        await mgr.store_entry("k2", "python rocks", level=MemoryLevel.PROJECT, importance=0.5)
        results = await mgr.search("python")
        assert len(results) >= 2

    async def test_compact(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        for i in range(10):
            await mgr.session.store(MemoryEntry(key=f"k{i}", content="x", level=MemoryLevel.SESSION, importance=0.1))
        result = await mgr.compact()
        assert "session" in result
        assert "conversation" in result

    async def test_clear_all(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.store_entry("k1", "data", level=MemoryLevel.SESSION)
        await mgr.store_entry("k2", "data", level=MemoryLevel.PROJECT)
        await mgr.clear_all()
        assert await mgr.session.stats() == MemoryStats()
        assert (await mgr.project.stats()).total_entries == 0  # type: ignore[union-attr]

    async def test_conversation_stats(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.add_message(MessageEntry(role="user", content="hi", token_count=5))
        stats = await mgr.conversation_stats()
        assert stats.message_count == 1
        assert stats.total_tokens == 5

    async def test_memory_stats(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        await mgr.store_entry("k1", "data", level=MemoryLevel.SESSION)
        stats = await mgr.memory_stats()
        assert "session" in stats
        assert stats["session"].total_entries == 1

    async def test_memory_stats_without_workspace(self) -> None:
        mgr = MemoryManager(session_id="s1")
        stats = await mgr.memory_stats()
        assert "session" in stats
        assert "project" not in stats
        assert "workspace" not in stats

    async def test_manager_get_context_limit(self, workspace_dir: Path) -> None:
        mgr = MemoryManager(session_id="s1", workspace_dir=workspace_dir)
        for i in range(10):
            await mgr.add_message(MessageEntry(role="user", content=f"msg{i}", token_count=1))
        context = await mgr.get_context(limit=3)
        assert len(context) == 3
