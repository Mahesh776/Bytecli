from bytecli.commands.base import Command, CommandContext
from bytecli.tools.registry import ToolRegistry


class ToolsCommand(Command):
    name = "tools"
    description = "List all available tools"
    category = "Tools"
    usage = "/tools"

    async def execute(self, args: str, context: CommandContext) -> bool:
        tools = ToolRegistry.list_tools()
        if not tools:
            context.renderer.info("No tools registered.")
            return True
        items = [(t.name, t.description) for t in tools]
        context.renderer.help_table(items, title="Tools")
        return True


class ToolCommand(Command):
    name = "tool"
    description = "Show details for a specific tool"
    category = "Tools"
    usage = "/tool <name>"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if not args:
            context.renderer.error("Usage: /tool <name>")
            return True
        tool = ToolRegistry.get(args)
        if tool is None:
            context.renderer.error(f"Unknown tool: {args}")
            return True
        schema = tool.get_json_schema()
        lines = [
            f"[bold]{tool.name}[/]",
            f"  Description: {tool.description}",
            f"  Parameters: {len(schema.get('properties', {}))}",
            f"  Required: {', '.join(schema.get('required', [])) or 'none'}",
        ]
        context.renderer.info("\n".join(lines))
        return True
