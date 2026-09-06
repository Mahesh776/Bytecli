from bytecli import __version__
from bytecli.commands.base import Command, CommandContext


class HelpCommand(Command):
    name = "help"
    aliases = ["h", "?"]  # noqa: RUF012
    description = "Show this help message with all commands"
    category = "Session"
    usage = "/help [command]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        registry = getattr(context.session, "_command_registry", None) if context.session else None
        if registry is None:
            context.renderer.info("No command registry available.")
            return True

        if args:
            cmd = registry.get(args)
            if cmd is None:
                context.renderer.error(f"Unknown command: {args}")
                return True
            lines = [
                f"[bold]{cmd.name}[/]",
                f"  Description: {cmd.description}",
                f"  Category: {cmd.category}",
                f"  Usage: {cmd.usage or '/' + cmd.name}",
            ]
            if cmd.aliases:
                lines.append(f"  Aliases: {', '.join(cmd.aliases)}")
            context.renderer.info("\n".join(lines))
            return True

        categories = registry.list_by_category()
        for cat, cmds in categories:
            items = [(f"/{c.name}", c.description) for c in cmds]
            context.renderer.help_table(items, title=cat)
        return True


class ExitCommand(Command):
    name = "exit"
    aliases = ["quit", "q"]  # noqa: RUF012
    description = "Exit the REPL"
    category = "Session"
    usage = "/exit"

    async def execute(self, args: str, context: CommandContext) -> bool:
        return False


class ClearCommand(Command):
    name = "clear"
    description = "Clear conversation context"
    category = "Session"
    usage = "/clear"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if context.memory is not None:
            await context.memory.conversation.clear()
            context.renderer.info("Conversation context cleared.")
        else:
            context.renderer.info("No memory manager available.")
        return True


class HistoryCommand(Command):
    name = "history"
    description = "Show or clear command history"
    category = "Session"
    usage = "/history [clear]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args == "clear":

            if context.session is not None:
                hist = getattr(context.session, "_session", None)
                if hist is not None:
                    context.renderer.info("Session history cleared.")
            return True
        context.renderer.info(f"ByteCli v{__version__} - type /history clear to clear history.")
        return True
