import asyncio
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.styles import Style

from bytecli import __version__
from bytecli.cli.commands import SlashCommandHandler
from bytecli.cli.completer import ByteCliCompleter
from bytecli.cli.rendering import RichRenderer
from bytecli.core.agent import AgentLoop
from bytecli.memory.manager import MemoryManager
from bytecli.tools.registry import ToolRegistry

STYLE = Style.from_dict({
    "prompt": "bold cyan",
    "input": "white",
})


class ReplSession:
    def __init__(
        self,
        agent: AgentLoop,
        memory: MemoryManager | None = None,
        renderer: RichRenderer | None = None,
        history_file: Path | None = None,
        plugin_manager: Any | None = None,
        skill_manager: Any | None = None,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._renderer = renderer or RichRenderer()
        self._command_handler = SlashCommandHandler(
            self._renderer, self._agent, self._memory, session=self,
            plugin_manager=plugin_manager, skill_manager=skill_manager,
        )
        self._running = True

        tool_names = [t.name for t in ToolRegistry.list_tools()]
        completer = ByteCliCompleter(tool_names=tool_names)

        history = FileHistory(str(history_file)) if history_file else None

        bindings = KeyBindings()

        @bindings.add("c-c")
        def _on_ctrl_c(event: KeyPressEvent) -> None:
            self._running = False
            event.app.exit()

        @bindings.add("c-d")
        def _on_ctrl_d(event: KeyPressEvent) -> None:
            event.app.exit()

        self._session: PromptSession[str] = PromptSession(
            history=history,
            completer=completer,
            complete_while_typing=True,
            key_bindings=bindings,
            style=STYLE,
        )

    @property
    def renderer(self) -> RichRenderer:
        return self._renderer

    async def run(self) -> int:
        try:
            await self._show_welcome()
            await self._main_loop()
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self._renderer.info("Goodbye!")
        return 0

    async def _show_welcome(self) -> None:
        self._renderer.welcome(
            version=__version__,
            provider=self._agent.config.model,
            model=self._agent.config.model,
        )

    async def _main_loop(self) -> None:
        while self._running:
            try:
                user_input = await self._session.prompt_async(
                    "bytecli> ",
                    bottom_toolbar=self._get_status_bar,
                )
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._running = await self._command_handler.handle(user_input)
                continue

            self._renderer.user_message(user_input)

            try:
                output = await self._agent.run(user_input)

                if output.response.content:
                    self._renderer.assistant_message(output.response.content)

                for turn in output.turns:
                    for tc in turn.tool_calls:
                        name = tc.function.get("name", "")
                        args = tc.function.get("arguments", "")
                        self._renderer.tool_call(name, {"args": args})

                    for tr in turn.tool_results:
                        tool_name = tr.name or "tool"
                        self._renderer.tool_result(tool_name, "error" not in (tr.content or "").lower(), tr.content)

                self._renderer.separator()

            except Exception as e:
                self._renderer.error(str(e))

    def _get_status_bar(self) -> str:
        parts = [f"[bold]ByteCli[/] v{__version__}"]
        parts.append(f"model: [cyan]{self._agent.config.model}[/]")
        parts.append(f"turns: [yellow]{self._agent.config.max_turns}[/]")
        return " | ".join(parts)
