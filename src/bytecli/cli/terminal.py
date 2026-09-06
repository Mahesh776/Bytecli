from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import has_focus
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import CompletionsMenu, Float, FloatContainer
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from bytecli import __version__
from bytecli.cli.commands import SlashCommandHandler
from bytecli.cli.completer import ByteCliCompleter
from bytecli.cli.rendering import RichRenderer
from bytecli.core.agent import AgentLoop
from bytecli.memory.manager import MemoryManager
from bytecli.tools.registry import ToolRegistry

BYTECLI_LOGO = (
    "██████╗ ██╗   ██╗████████╗███████╗ ██████╗██╗     ██╗\n"
    "██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔════╝██╔════╝██║     ██║\n"
    "██████╔╝ ╚████╔╝    ██║   █████╗  ██║     ██║     ██║\n"
    "██╔══██╗  ╚██╔╝     ██║   ██╔══╝  ██║     ██║     ██║\n"
    "██████╔╝   ██║      ██║   ███████╗╚██████╗███████╗██║\n"
    "╚═════╝    ╚═╝      ╚═╝   ╚══════╝ ╚═════╝╚══════╝╚═╝"
)

LOGO_LINES = BYTECLI_LOGO.split("\n")
LOGO_HEIGHT = len(LOGO_LINES)
LOGO_WIDTH = max(len(line) for line in LOGO_LINES)

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

LOGO_GRADIENT = ("#4A9EFF", "#5B8CFF", "#7C6CFA", "#8B5CF6", "#A855F7", "#D946EF")

ACCENT_BLUE = "#4A9EFF"
ACCENT_VIOLET = "#7C6CFA"
ACCENT_PURPLE = "#A855F7"
ACCENT_FUCHSIA = "#D946EF"
ACCENT_GREEN = "#22C55E"
ACCENT_GRAY = "#6B7280"
ACCENT_DIM = "#9CA3AF"
TEXT_MAIN = "#E5E7EB"

TERMINAL_STYLE = Style.from_dict(
    {
        "frame": "bg:#0d0d12",
        "frame.border": ACCENT_GRAY,
        "frame_header frame.border": ACCENT_PURPLE,
        "frame_info frame.border": ACCENT_BLUE,
        "frame_conversation frame.border": ACCENT_VIOLET,
        "frame_input frame.border": ACCENT_BLUE,
        "frame_footer frame.border": ACCENT_GRAY,
        "info.label": ACCENT_DIM,
        "footer.right": ACCENT_GRAY,
        "input.prompt": f"bold {ACCENT_PURPLE}",
        "text-area.prompt": f"bold {ACCENT_PURPLE}",
        "text-area": TEXT_MAIN,
        "input": TEXT_MAIN,
        "completion-menu": f"bg:#1e1b2e fg:{TEXT_MAIN}",
        "completion-menu.completion": f"bg:#1e1b2e fg:{TEXT_MAIN}",
        "completion-menu.completion.current": f"bg:{ACCENT_VIOLET} fg:#0d0d12",
        "completion-menu.meta": f"bg:#16131f fg:{ACCENT_DIM}",
        "completion-menu.meta.completion": f"bg:{ACCENT_VIOLET} fg:#0d0d12",
        "completion-menu.meta.completion.current": f"bg:#2a2540 fg:{TEXT_MAIN}",
        "completion-menu.progress": f"bg:{ACCENT_BLUE}",
        "error.copy": f"bg:#2a2540 fg:{ACCENT_FUCHSIA} underline",
    }
)


class TerminalConsole(RichRenderer):
    def __init__(self, terminal: ByteCliTerminal) -> None:
        super().__init__()
        self._terminal = terminal
        self._buffer = io.StringIO()
        self._console = Console(
            file=self._buffer,
            force_terminal=True,
            color_system="256",
            width=self._width(),
        )

    @property
    def console(self) -> Console:
        return self._console

    def _width(self) -> int:
        app = self._terminal.app
        if app is not None:
            try:
                size = app.output.get_size()
                return max(40, size.columns - 4)
            except Exception:
                pass
        return 84

    def _reset(self) -> None:
        self._buffer = io.StringIO()
        self._console = Console(
            file=self._buffer,
            force_terminal=True,
            color_system="256",
            width=self._width(),
        )

    def _flush(self) -> None:
        ansi = self._buffer.getvalue()
        self._reset()
        if ansi.strip():
            self._terminal.append_block(ansi)

    def welcome(self, version: str, provider: str, model: str) -> None:
        text = Text()
        text.append("Welcome to ByteCli AI Agent ", style="bold " + ACCENT_VIOLET)
        text.append(f"v{version}", style=ACCENT_FUCHSIA)
        text.append("\nProvider: ", style="dim")
        text.append(provider, style=ACCENT_BLUE)
        text.append("  |  Model: ", style="dim")
        text.append(model, style=ACCENT_BLUE)
        self._console.print(Panel(text, border_style=ACCENT_VIOLET))
        self._flush()
        self._console.print(
            f"[dim]Tips:[/] [bold {ACCENT_PURPLE}]/help[/] for commands, [bold {ACCENT_PURPLE}]/exit[/] to quit, "
            f"[bold {ACCENT_PURPLE}]Ctrl+C[/] to interrupt a running task"
        )
        self._flush()

    def user_message(self, content: str) -> None:
        self._console.print(f"[bold {ACCENT_PURPLE}]You >[/] {content}")
        self._flush()

    def assistant_message(self, content: str) -> None:
        markdown = RichMarkdown(content)
        self._console.print(f"[bold {ACCENT_BLUE}]AI >[/]")
        self._console.print(markdown)
        self._flush()

    def assistant_streaming(self, content: str) -> None:
        self._console.print(content, end="")

    def tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        panel = Panel(
            f"[bold {ACCENT_PURPLE}]{tool_name}[/]",
            title=f"[{ACCENT_FUCHSIA}]Tool Call[/]",
            border_style=ACCENT_VIOLET,
            subtitle=f"Args: {args}",
        )
        self._console.print(panel)
        self._flush()

    def tool_result(self, tool_name: str, success: bool, data: Any) -> None:
        style = "green" if success else "red"
        label = "Success" if success else "Error"
        content = str(data)[:500]
        panel = Panel(content, title=f"[{style}]{label}[/] - {tool_name}", border_style=style)
        self._console.print(panel)
        self._flush()
        if not success:
            self._terminal.append_copy_button(str(data))

    def error(self, message: str) -> None:
        self._console.print(f"\n[bold red]Error:[/] {message}")
        self._flush()
        self._terminal.append_copy_button(message)

    def info(self, message: str) -> None:
        self._console.print(f"\n[bold {ACCENT_PURPLE}]Info:[/] {message}")
        self._flush()

    def warning(self, message: str) -> None:
        self._console.print(f"\n[bold yellow]Warning:[/] {message}")
        self._flush()

    def help_table(self, commands: list[tuple[str, str]], title: str = "Commands") -> None:
        table = Table(title=title, border_style=ACCENT_VIOLET)
        table.add_column("Command", style=f"bold {ACCENT_PURPLE}")
        table.add_column("Description", style=TEXT_MAIN)
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self._console.print(table)
        self._flush()

    def memory_stats_table(self, stats: dict[str, Any]) -> None:
        table = Table(title="Memory Stats", border_style=ACCENT_FUCHSIA)
        table.add_column("Store", style="bold")
        table.add_column("Entries", style=ACCENT_BLUE)
        table.add_column("Tokens", style=ACCENT_PURPLE)
        for name, s in stats.items():
            table.add_row(name, str(getattr(s, "total_entries", "?")), str(getattr(s, "total_tokens", "?")))
        self._console.print(table)
        self._flush()

    def code_block(self, code: str, language: str = "python") -> None:
        syntax = Syntax(code, language, theme=self._theme, line_numbers=True)
        self._console.print(syntax)
        self._flush()

    def separator(self) -> None:
        self._console.print(Rule(style="dim"))
        self._flush()

    def status(self, text: str = "Working...") -> None:
        self._console.print(f"[dim]{text}[/]")
        self._flush()


class ByteCliTerminal:
    def __init__(
        self,
        agent: AgentLoop,
        memory: MemoryManager | None = None,
        history_file: Path | None = None,
        plugin_manager: Any | None = None,
        skill_manager: Any | None = None,
        provider_name: str | None = None,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._provider_name = provider_name or "Local"
        self._plugin_manager = plugin_manager
        self._skill_manager = skill_manager
        self._app: Application[None] | None = None

        self._console = TerminalConsole(self)
        self._command_handler = SlashCommandHandler(
            self._console,
            self._agent,
            self._memory,
            session=self,
            plugin_manager=plugin_manager,
            skill_manager=skill_manager,
        )
        self._command_registry = self._command_handler.registry

        self._running = True
        self._thinking = False
        self._spinner_index = 0
        self._spinner_task: asyncio.Task[Any] | None = None
        self._current_task: asyncio.Task[Any] | None = None
        self._blocks: list[StyleAndTextTuples] = []
        self._resolved_model: str | None = None
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        tool_names = [t.name for t in ToolRegistry.list_tools()]
        completer = ByteCliCompleter(tool_names=tool_names)
        history = FileHistory(str(history_file)) if history_file else None

        self._input = TextArea(
            prompt=[("class:input.prompt", "> ")],
            multiline=False,
            completer=completer,
            complete_while_typing=True,
            history=history,
            accept_handler=self._on_accept,
            style="class:input",
        )

        self._bindings = KeyBindings()
        self._setup_bindings()
        self._conversation_window = self._build_conversation_window()

    def _build_app(self) -> Application[None]:
        return Application(
            layout=Layout(
                FloatContainer(
                    self._root_container(),
                    floats=[
                        Float(
                            xcursor=True,
                            ycursor=True,
                            transparent=True,
                            content=CompletionsMenu(
                                max_height=12,
                                scroll_offset=1,
                                extra_filter=has_focus(self._input.buffer),
                            ),
                        ),
                    ],
                ),
                focused_element=self._input.window,
            ),
            style=TERMINAL_STYLE,
            full_screen=True,
            key_bindings=self._bindings,
            mouse_support=True,
        )

    @property
    def app(self) -> Application[None] | None:
        return self._app

    @property
    def blocks(self) -> list[StyleAndTextTuples]:
        return self._blocks

    @property
    def renderer(self) -> TerminalConsole:
        return self._console

    def _setup_bindings(self) -> None:
        @self._bindings.add("c-c")
        def _on_ctrl_c(event: KeyPressEvent) -> None:
            if self._thinking and self._current_task is not None:
                self._current_task.cancel()
            else:
                event.app.exit()

        @self._bindings.add("c-d")
        def _on_ctrl_d(event: KeyPressEvent) -> None:
            event.app.exit()

    def _build_conversation_window(self) -> Window:
        return Window(
            FormattedTextControl(
                self._get_conversation_text,
                get_cursor_position=self._get_conversation_cursor_position,
            ),
            wrap_lines=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin()],
        )

    def _root_container(self) -> HSplit:
        return HSplit(
            [
                Frame(
                    VSplit(
                        [
                            Window(
                                FormattedTextControl(self._get_header_logo),
                                width=LOGO_WIDTH,
                                height=LOGO_HEIGHT,
                            ),
                            Window(
                                FormattedTextControl(self._get_header_status),
                                align=WindowAlign.RIGHT,
                                height=LOGO_HEIGHT,
                            ),
                        ]
                    ),
                    style="class:frame_header",
                    height=LOGO_HEIGHT + 2,
                ),
                Frame(
                    Window(FormattedTextControl(self._get_info_fragments), height=1),
                    style="class:frame_info",
                    height=3,
                ),
                Frame(
                    self._conversation_window,
                    style="class:frame_conversation",
                ),
                Frame(self._input.window, style="class:frame_input", height=3),
                Frame(self._footer_container(), style="class:frame_footer", height=3),
            ]
        )

    def _footer_container(self) -> VSplit:
        return VSplit(
            [
                Window(FormattedTextControl(self._get_footer_left), height=1),
                Window(FormattedTextControl(self._get_footer_right), height=1, align=WindowAlign.RIGHT),
            ]
        )

    def _get_header_logo(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for i, line in enumerate(LOGO_LINES):
            if i:
                fragments.append(("", "\n"))
            fragments.append((f"bold {LOGO_GRADIENT[i]}", line))
        return fragments

    def _get_header_status(self) -> StyleAndTextTuples:
        memory_state = "Memory ON" if self._memory is not None else "Memory OFF"
        tool_count = len(ToolRegistry.list_tools())
        memory_color = ACCENT_GREEN if self._memory is not None else ACCENT_GRAY
        return [
            ("bold " + ACCENT_PURPLE, f"v{__version__}"),
            ("", "  "),
            ("bold " + ACCENT_GREEN, "Ready ●"),
            ("", "\n"),
            (memory_color, memory_state),
            ("", "  |  "),
            (ACCENT_BLUE, f"Tools {tool_count}"),
            ("", "\n"),
            (ACCENT_VIOLET, "Coding Mode"),
        ]

    def _get_info_fragments(self) -> StyleAndTextTuples:
        model = self._resolved_model or self._agent.config.model
        context = str(self._agent.config.max_tokens)
        provider = self._provider_name
        return [
            ("class:info.label", " Model : "),
            (ACCENT_BLUE, model),
            ("class:info.label", "  |  Context : "),
            (ACCENT_PURPLE, context),
            ("class:info.label", "  |  Provider : "),
            (ACCENT_BLUE, provider),
            ("class:info.label", "  |  Session : "),
            (ACCENT_GRAY, "Local"),
        ]

    def _get_footer_left(self) -> StyleAndTextTuples:
        if self._thinking:
            frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            return [("bold " + ACCENT_FUCHSIA, f"{frame} Thinking..."), (ACCENT_GRAY, "   Press Ctrl+C to stop")]
        return [("bold " + ACCENT_GREEN, "Ready ●")]

    def _get_footer_right(self) -> StyleAndTextTuples:
        used = self._total_prompt_tokens + self._total_completion_tokens
        context = str(self._agent.config.max_tokens)
        model = self._resolved_model or self._agent.config.model
        text = f"Model: {model}  |  Tokens: {used}/{context}  |  ByteCli v{__version__}"
        return [("class:footer.right", text)]

    def _get_conversation_text(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for i, block in enumerate(self._blocks):
            if i:
                fragments.append(("", "\n\n"))
            fragments.extend(block)
        return fragments

    def _get_conversation_cursor_position(self) -> Point:
        fragments = self._get_conversation_text()
        last_line = sum(fragment[1].count("\n") for fragment in fragments)
        return Point(x=0, y=last_line)

    def _invalidate(self) -> None:
        self._conversation_window.vertical_scroll = 10**9
        if self._app is not None:
            self._app.invalidate()

    def append_block(self, ansi_text: str) -> None:
        fragments = ANSI(ansi_text.rstrip()).__pt_formatted_text__()
        if fragments:
            self._blocks.append(fragments)
            self._invalidate()

    def append_copy_button(self, text: str) -> None:
        handler = self._make_copy_handler(text)
        self._blocks.append(
            [
                ("", "  "),
                ("class:error.copy", "[ Copy Error ]", handler),
                ("", ""),
            ]
        )
        self._invalidate()

    def _make_copy_handler(self, text: str) -> Callable[[MouseEvent], object]:
        def _on_click(event: MouseEvent) -> object:
            if event.event_type == MouseEventType.MOUSE_UP and self._app is not None:
                if self._copy_to_clipboard(text):
                    self._console.info("Error copied to clipboard")
                else:
                    self._console.warning("Copy to system clipboard failed")
            return None

        return _on_click

    def _copy_to_clipboard(self, text: str) -> bool:
        try:
            import pyperclip

            pyperclip.copy(text)
            return True
        except Exception:
            pass
        if self._app is None:
            return False
        try:
            self._app.clipboard.set_data(ClipboardData(text))
            return True
        except Exception:
            return False

    def clear_conversation(self) -> None:
        self._blocks.clear()
        self._invalidate()

    def _on_accept(self, buffer: Buffer) -> bool:
        text = buffer.text
        if self._thinking or not text.strip():
            return True
        if self._app is None:
            return True
        self._thinking = True
        self._current_task = self._app.create_background_task(self._handle_input(text))
        return False

    async def run(self) -> int:
        app = self._build_app()
        self._app = app
        try:
            self._resolved_model = await self._agent.resolved_model()
            self._console.welcome(
                version=__version__,
                provider=self._provider_name,
                model=self._resolved_model,
            )
            await app.run_async()
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._stop_spinner()
            self._app = None
        return 0

    async def _handle_input(self, text: str) -> None:
        try:
            if text.startswith("/"):
                self._thinking = False
                self._invalidate()
                await self._handle_command(text)
                return
            await self._run_agent(text)
        finally:
            self._thinking = False
            self._stop_spinner()
            self._current_task = None
            self._invalidate()

    async def _handle_command(self, command: str) -> None:
        name = command.strip().split(maxsplit=1)[0]
        if name in ("/clear",):
            self.clear_conversation()
        try:
            keep_running = await self._command_handler.handle(command)
        except Exception as exc:
            self._console.error(str(exc))
            return
        if not keep_running:
            self._running = False
            if self._app is not None and self._app.future is not None:
                self._app.exit()

    async def _run_agent(self, text: str) -> None:
        self._console.user_message(text)
        self._start_spinner()
        self._invalidate()
        try:
            output = await self._agent.run(text)

            if output.truncated and output.error:
                self._console.warning(output.error)

            if output.response.content:
                self._console.assistant_message(output.response.content)

            for turn in output.turns:
                for tc in turn.tool_calls:
                    name = tc.function.get("name", "")
                    args = tc.function.get("arguments", "")
                    self._console.tool_call(name, {"args": args})

                for tr in turn.tool_results:
                    tool_name = tr.name or "tool"
                    success = "error" not in (tr.content or "").lower()
                    self._console.tool_result(tool_name, success, tr.content)

            self._total_prompt_tokens += output.stats.total_prompt_tokens
            self._total_completion_tokens += output.stats.total_completion_tokens
        except asyncio.CancelledError:
            self._console.error("Task cancelled.")
        except Exception as exc:
            self._console.error(str(exc))

    def _start_spinner(self) -> None:
        self._spinner_index = 0
        loop = asyncio.get_running_loop()
        self._spinner_task = loop.create_task(self._animate_spinner())

    def _stop_spinner(self) -> None:
        if self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None

    async def _animate_spinner(self) -> None:
        try:
            while True:
                self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
                self._invalidate()
                await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            pass
