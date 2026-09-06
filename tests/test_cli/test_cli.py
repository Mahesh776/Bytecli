from pathlib import Path

import pytest

from bytecli.cli.commands import SlashCommandHandler
from bytecli.cli.completer import ByteCliCompleter
from bytecli.cli.rendering import RichRenderer
from bytecli.core.agent import AgentLoop
from bytecli.core.turn import AgentConfig
from bytecli.memory.manager import MemoryManager
from bytecli.providers.base import Provider
from bytecli.providers.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    ModelInfo,
    Role,
    Usage,
)
from bytecli.tools.registry import ToolRegistry


class FakeCliProvider(Provider):
    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            id="r1",
            model="test",
            choices=[Choice(index=0, message=Message(role=Role.ASSISTANT, content="Hello!"))],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    def chat_stream(self, request: CompletionRequest):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="test")]

    def _build_headers(self) -> dict[str, str]:
        return {}

    def _build_chat_payload(self, request: CompletionRequest) -> dict:
        return {}

    def _parse_response(self, data: dict) -> CompletionResponse:
        raise NotImplementedError

    def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
        return None


@pytest.fixture
def agent() -> AgentLoop:
    return AgentLoop(provider=FakeCliProvider(), config=AgentConfig(model="test"))


@pytest.fixture
def memory() -> MemoryManager:
    return MemoryManager(session_id="test-session")


@pytest.fixture
def renderer() -> RichRenderer:
    return RichRenderer()


class TestRichRenderer:
    def test_console(self, renderer: RichRenderer) -> None:
        assert renderer.console is not None

    def test_error(self, renderer: RichRenderer) -> None:
        renderer.error("test error")

    def test_info(self, renderer: RichRenderer) -> None:
        renderer.info("test info")

    def test_separator(self, renderer: RichRenderer) -> None:
        renderer.separator()

    def test_status(self, renderer: RichRenderer) -> None:
        renderer.status("working")


class TestByteCliCompleter:
    @pytest.fixture
    def completer(self) -> ByteCliCompleter:
        return ByteCliCompleter(tool_names=["read", "write", "bash"])

    def test_complete_command_help(self, completer: ByteCliCompleter) -> None:
        from prompt_toolkit.document import Document

        doc = Document("/h")
        completions = completer.get_completions(doc, None)  # type: ignore[arg-type]
        texts = [c.text for c in completions]
        assert any("/help" in t for t in texts)

    def test_complete_command_exit(self, completer: ByteCliCompleter) -> None:
        from prompt_toolkit.document import Document

        doc = Document("/e")
        completions = completer.get_completions(doc, None)  # type: ignore[arg-type]
        texts = [c.text for c in completions]
        assert any("/exit" in t for t in texts)

    def test_no_completion_for_empty_slash(self, completer: ByteCliCompleter) -> None:
        from prompt_toolkit.document import Document

        doc = Document("/")
        completions = completer.get_completions(doc, None)  # type: ignore[arg-type]
        assert len(completions) >= len(["/help", "/exit", "/quit"])

    def test_complete_tool(self, completer: ByteCliCompleter) -> None:
        from prompt_toolkit.document import Document

        doc = Document("re")
        completions = completer.get_completions(doc, None)  # type: ignore[arg-type]
        texts = [c.text for c in completions]
        assert any("read" in t for t in texts)

    def test_no_completion_for_non_matching(self, completer: ByteCliCompleter) -> None:
        from prompt_toolkit.document import Document

        doc = Document("zzz")
        completions = completer.get_completions(doc, None)  # type: ignore[arg-type]
        assert len(completions) == 0

    def test_update_tool_names(self) -> None:
        c = ByteCliCompleter()
        c.update_tool_names(["foo", "bar"])
        from prompt_toolkit.document import Document

        doc = Document("f")
        completions = c.get_completions(doc, None)  # type: ignore[arg-type]
        texts = [c.text for c in completions]
        assert "foo" in texts


class TestSlashCommandHandler:
    @pytest.fixture
    def handler(self, agent: AgentLoop, memory: MemoryManager) -> SlashCommandHandler:
        return SlashCommandHandler(RichRenderer(), agent, memory)

    async def test_help(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/help")
        assert result is True

    async def test_exit(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/exit")
        assert result is False

    async def test_quit(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/quit")
        assert result is False

    async def test_clear(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/clear")
        assert result is True

    async def test_model(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/model")
        assert result is True

    async def test_system_set(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/system Be helpful")
        assert result is True

    async def test_system_show(self, handler: SlashCommandHandler) -> None:
        handler._agent.config.system_prompt = "You are helpful."
        result = await handler.handle("/system")
        assert result is True

    async def test_memory(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/memory")
        assert result is True

    async def test_tokens(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/tokens")
        assert result is True

    async def test_compact(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/compact")
        assert result is True

    async def test_tools(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/tools")
        assert result is True

    async def test_unknown_command(self, handler: SlashCommandHandler) -> None:
        result = await handler.handle("/unknown")
        assert result is True

    async def test_handler_without_memory(self, agent: AgentLoop) -> None:
        handler = SlashCommandHandler(RichRenderer(), agent, memory=None)
        result = await handler.handle("/memory")
        assert result is True

    async def test_handler_clear_without_memory(self, agent: AgentLoop) -> None:
        handler = SlashCommandHandler(RichRenderer(), agent, memory=None)
        result = await handler.handle("/clear")
        assert result is True
