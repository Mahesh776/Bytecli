
from bytecli.commands.base import Command, CommandContext
from bytecli.commands.session import (
    ClearCommand,
    ExitCommand,
    HelpCommand,
    HistoryCommand,
)


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._by_category: dict[str, list[Command]] = {}

    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command
        self._by_category.setdefault(command.category, []).append(command)

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def list_commands(self) -> list[Command]:
        seen: set[str] = set()
        result: list[Command] = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd)
        result.sort(key=lambda c: c.name)
        return result

    def list_by_category(self) -> list[tuple[str, list[Command]]]:
        return sorted(self._by_category.items(), key=lambda x: x[0])

    async def execute(self, name: str, args: str, context: CommandContext) -> bool:
        cmd = self.get(name)
        if cmd is None:
            context.renderer.error(f"Unknown command: /{name}. Type /help for available commands.")
            return True
        return await cmd.execute(args, context)

    @property
    def commands(self) -> dict[str, Command]:
        return self._commands


def register_all() -> CommandRegistry:
    registry = CommandRegistry()

    from bytecli.commands.agent import RetryCommand, StatusCommand, UndoCommand
    from bytecli.commands.config import (
        ConfigCommand,
        MaxTokensCommand,
        ModelCommand,
        ProviderCommand,
        SystemCommand,
        TemperatureCommand,
        TopPCommand,
    )
    from bytecli.commands.diagnostics import (
        DebugCommand,
        EnvCommand,
        LogCommand,
        PluginCommand,
        PluginsCommand,
        SkillCommand,
        SkillsCommand,
        VersionCommand,
    )
    from bytecli.commands.export_import import ExportCommand, ImportCommand
    from bytecli.commands.memory import (
        CompactCommand,
        ContextCommand,
        ForgetCommand,
        MemoryCommand,
        TokensCommand,
    )
    from bytecli.commands.tools import ToolCommand, ToolsCommand
    from bytecli.commands.workspace import (
        FrameworksCommand,
        GitCommand,
        LanguageCommand,
        TreeCommand,
        WorkspaceCommand,
    )

    commands = [
        HelpCommand(),
        ExitCommand(),
        ClearCommand(),
        HistoryCommand(),
        ModelCommand(),
        ProviderCommand(),
        TemperatureCommand(),
        SystemCommand(),
        MaxTokensCommand(),
        TopPCommand(),
        ConfigCommand(),
        MemoryCommand(),
        TokensCommand(),
        CompactCommand(),
        ForgetCommand(),
        ContextCommand(),
        ToolsCommand(),
        ToolCommand(),
        WorkspaceCommand(),
        TreeCommand(),
        GitCommand(),
        LanguageCommand(),
        FrameworksCommand(),
        StatusCommand(),
        RetryCommand(),
        UndoCommand(),
        LogCommand(),
        DebugCommand(),
        VersionCommand(),
        EnvCommand(),
        PluginsCommand(),
        PluginCommand(),
        SkillsCommand(),
        SkillCommand(),
        ExportCommand(),
        ImportCommand(),
    ]

    for cmd in commands:
        registry.register(cmd)

    return registry
