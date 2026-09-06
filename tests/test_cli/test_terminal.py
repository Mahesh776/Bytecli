import asyncio
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from bytecli.cli.terminal import ByteCliTerminal
from bytecli.core.turn import AgentOutput, TurnStats
from bytecli.providers.types import Message, Role


class FakeAgent:
    def __init__(self, response: str = "Hello there!") -> None:
        self.config = SimpleNamespace(model="test-model", max_tokens=1024)
        self._response = response
        self.calls: list[str] = []

    async def run(self, text: str) -> AgentOutput:
        self.calls.append(text)
        return AgentOutput(
            messages=[],
            response=Message(role=Role.ASSISTANT, content=self._response),
            turns=[],
            stats=TurnStats(total_prompt_tokens=4, total_completion_tokens=2),
        )


class FakeApp:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[Any]] = []
        self.invalidations = 0

    def create_background_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self.tasks.append(task)
        return task

    def invalidate(self) -> None:
        self.invalidations += 1


@pytest.fixture
def terminal() -> ByteCliTerminal:
    return ByteCliTerminal(FakeAgent(), provider_name="LM Studio")


def _joined(block: list[tuple[str, str]]) -> str:
    return "".join(item[1] for item in block)


def test_header_logo_has_consistent_width(terminal: ByteCliTerminal) -> None:
    logo = terminal._get_header_logo()
    lines = [text for style, text in logo if text != "\n"]
    assert len(lines) == 6
    assert len({len(line) for line in lines}) == 1
    assert len({style for style, _ in logo}) == 7


def test_build_app_layout_has_completion_menu(terminal: ByteCliTerminal) -> None:
    import prompt_toolkit.output.defaults as od
    from prompt_toolkit.layout import CompletionsMenu, FloatContainer
    from prompt_toolkit.output import DummyOutput

    original = od.create_output
    od.create_output = lambda *a, **k: DummyOutput()
    try:
        app = terminal._build_app()
        root = app.layout.container
        assert isinstance(root, FloatContainer)
        contents = [f.content for f in root.floats]
        assert any(isinstance(c, CompletionsMenu) for c in contents)
    finally:
        od.create_output = original


def test_conversation_cursor_follows_last_line(terminal: ByteCliTerminal) -> None:
    for i in range(10):
        terminal._console.user_message(f"message {i}")
    point = terminal._get_conversation_cursor_position()
    text = "".join(text for _, text in terminal._get_conversation_text())
    assert point.y == text.count("\n")
    assert point.y >= 0


def test_conversation_window_scrolls_to_bottom(terminal: ByteCliTerminal) -> None:
    for i in range(40):
        terminal._console.user_message(f"short line {i}")
    window = terminal._build_conversation_window()
    control = window.content
    width, height = 60, 24
    ui_content = control.create_content(width, height)
    assert ui_content.cursor_position.y > height
    window.vertical_scroll = 10**9
    window._scroll_when_linewrapping(ui_content, width, height)
    assert window.vertical_scroll == max(0, ui_content.line_count - height)


def test_conversation_window_without_cursor_is_stuck_at_top(terminal: ByteCliTerminal) -> None:
    for i in range(40):
        terminal._console.user_message(f"short line {i}")
    window = Window(FormattedTextControl(terminal._get_conversation_text), wrap_lines=True)
    ui_content = window.content.create_content(60, 24)
    window.vertical_scroll = 10**9
    window._scroll_when_linewrapping(ui_content, 60, 24)
    assert window.vertical_scroll == 0


def test_info_fragments_show_model_and_provider(terminal: ByteCliTerminal) -> None:
    info = "".join(text for _, text in terminal._get_info_fragments())
    assert "test-model" in info
    assert "LM Studio" in info
    assert "Session : Local" in info


def test_info_fragments_prefer_resolved_model(terminal: ByteCliTerminal) -> None:
    terminal._resolved_model = "qwen2.5-coder-1.5b-instruct"
    info = "".join(text for _, text in terminal._get_info_fragments())
    assert "qwen2.5-coder-1.5b-instruct" in info
    assert "test-model" not in info


def test_footer_right_shows_resolved_model(terminal: ByteCliTerminal) -> None:
    terminal._resolved_model = "qwen2.5-coder-1.5b-instruct"
    footer = "".join(text for _, text in terminal._get_footer_right())
    assert "qwen2.5-coder-1.5b-instruct" in footer


@pytest.mark.asyncio
async def test_run_resolves_and_welcomes_with_model(terminal: ByteCliTerminal) -> None:
    import prompt_toolkit.application as pa
    import prompt_toolkit.output.defaults as od
    from prompt_toolkit.output import DummyOutput
    from unittest.mock import AsyncMock

    original_output = od.create_output
    od.create_output = lambda *a, **k: DummyOutput()
    original_run_async = pa.Application.run_async
    pa.Application.run_async = AsyncMock(return_value=None)

    class ResolvingAgent(FakeAgent):
        async def resolved_model(self) -> str:
            return "qwen2.5-coder-1.5b-instruct"

    term = ByteCliTerminal(ResolvingAgent(), provider_name="LM Studio")
    try:
        await term.run()
    finally:
        od.create_output = original_output
        pa.Application.run_async = original_run_async
    assert term._resolved_model == "qwen2.5-coder-1.5b-instruct"


def test_footer_right_shows_version_and_tokens(terminal: ByteCliTerminal) -> None:
    footer = "".join(text for _, text in terminal._get_footer_right())
    assert "ByteCli" in footer
    assert "Tokens: 0/1024" in footer


def test_footer_left_ready_when_idle(terminal: ByteCliTerminal) -> None:
    footer = "".join(text for _, text in terminal._get_footer_left())
    assert "Ready" in footer


@pytest.mark.asyncio
async def test_user_message_appends_block(terminal: ByteCliTerminal) -> None:
    terminal._console.user_message("hello world")
    assert terminal.blocks
    assert "You >" in _joined(terminal.blocks[-1])


@pytest.mark.asyncio
async def test_handle_input_runs_agent(terminal: ByteCliTerminal) -> None:
    await terminal._handle_input("hello world")
    joined = "".join(_joined(b) for b in terminal.blocks)
    assert "You > hello world" in joined
    assert "AI >" in joined
    assert "Hello there!" in joined
    assert terminal._total_prompt_tokens == 4
    assert terminal._total_completion_tokens == 2
    assert terminal._thinking is False


@pytest.mark.asyncio
async def test_handle_input_tracks_token_usage(terminal: ByteCliTerminal) -> None:
    await terminal._handle_input("one")
    await terminal._handle_input("two")
    assert terminal._total_prompt_tokens == 8
    assert terminal._total_completion_tokens == 4


@pytest.mark.asyncio
async def test_slash_help_renders_help_blocks(terminal: ByteCliTerminal) -> None:
    await terminal._handle_input("/help")
    assert terminal.blocks
    joined = "".join(_joined(b) for b in terminal.blocks)
    assert "Command" in joined
    assert "/exit" in joined


@pytest.mark.asyncio
async def test_slash_exit_stops_terminal(terminal: ByteCliTerminal) -> None:
    terminal._app = None
    await terminal._handle_input("/exit")
    assert terminal._running is False


@pytest.mark.asyncio
async def test_slash_clear_clears_conversation(terminal: ByteCliTerminal) -> None:
    terminal._console.user_message("keep me?")
    assert terminal.blocks
    await terminal._handle_input("/clear")
    joined = "".join(_joined(b) for b in terminal.blocks)
    assert "keep me?" not in joined


@pytest.mark.asyncio
async def test_on_accept_schedules_background_task(terminal: ByteCliTerminal) -> None:
    app = FakeApp()
    terminal._app = app  # type: ignore[assignment]
    terminal._input.text = "hello"
    result = terminal._on_accept(terminal._input.buffer)
    assert result is False
    assert terminal._thinking is True
    await asyncio.gather(*app.tasks)
    joined = "".join(_joined(b) for b in terminal.blocks)
    assert "You > hello" in joined
    assert terminal._thinking is False


def test_on_accept_ignores_empty_input(terminal: ByteCliTerminal) -> None:
    terminal._input.text = "   "
    assert terminal._on_accept(terminal._input.buffer) is True


def test_on_accept_ignores_input_while_thinking(terminal: ByteCliTerminal) -> None:
    terminal._thinking = True
    terminal._input.text = "queued"
    assert terminal._on_accept(terminal._input.buffer) is True
    assert terminal._input.text == "queued"


@pytest.mark.asyncio
async def test_tool_call_and_result_render_blocks(terminal: ByteCliTerminal) -> None:
    terminal._console.tool_call("read", {"args": "{}"})
    assert "read" in _joined(terminal.blocks[-1])
    terminal._console.tool_result("read", True, "file content")
    assert "read" in _joined(terminal.blocks[-1])
    terminal._console.tool_result("write", False, "boom")
    assert "write" in _joined(terminal.blocks[-2])
    assert "Copy Error" in _joined(terminal.blocks[-1])


@pytest.mark.asyncio
async def test_welcome_renders_panel(terminal: ByteCliTerminal) -> None:
    terminal._console.welcome("1.2.3", "LM Studio", "test-model")
    assert "ByteCli" in _joined(terminal.blocks[0])


@pytest.mark.asyncio
async def test_agent_error_is_reported(terminal: ByteCliTerminal) -> None:
    class BrokenAgent(FakeAgent):
        async def run(self, text: str) -> AgentOutput:
            raise RuntimeError("boom")

    broken = ByteCliTerminal(BrokenAgent(), provider_name="LM Studio")
    await broken._handle_input("do it")
    joined = "".join(_joined(b) for b in broken.blocks)
    assert "boom" in joined
    assert broken._thinking is False


def test_error_block_has_copy_button(terminal: ByteCliTerminal) -> None:
    from prompt_toolkit.clipboard import ClipboardData
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier

    class FakeClipboard:
        def __init__(self) -> None:
            self.data: ClipboardData | None = None

        def set_data(self, data: ClipboardData) -> None:
            self.data = data

    class ButtonApp:
        def __init__(self) -> None:
            self.clipboard = FakeClipboard()
            self.invalidations = 0

        def invalidate(self) -> None:
            self.invalidations += 1

    app = ButtonApp()
    terminal._app = app  # type: ignore[assignment]
    terminal._console.error("something went wrong")
    copy_block = terminal.blocks[-1]
    joined = _joined(copy_block)
    assert "Copy Error" in joined
    handler = next(item[2] for item in copy_block if len(item) == 3)
    event = MouseEvent(position=(0, 0), event_type=MouseEventType.MOUSE_UP, button=MouseButton.LEFT, modifiers=frozenset())

    with mock.patch("pyperclip.copy") as fake_copy:
        handler(event)
        fake_copy.assert_called_once_with("something went wrong")

    with mock.patch("pyperclip.copy", side_effect=RuntimeError("no clipboard")):
        handler(event)
        assert app.clipboard.data is not None
        assert "something went wrong" in app.clipboard.data.text
