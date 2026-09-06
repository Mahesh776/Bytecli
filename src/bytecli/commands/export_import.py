from bytecli.commands.base import Command, CommandContext


class ExportCommand(Command):
    name = "export"
    description = "Export conversation to JSON"
    category = "Export/Import"
    usage = "/export [filepath]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        msgs = await context.memory.get_context()
        data = [
            {"role": m.role, "content": m.content, "token_count": m.token_count}
            for m in msgs
        ]
        import json
        from pathlib import Path

        path = Path(args) if args else Path.cwd() / "conversation_export.json"
        path.write_text(json.dumps(data, indent=2), "utf-8")
        context.renderer.info(f"Exported {len(data)} messages to {path}")
        return True


class ImportCommand(Command):
    name = "import"
    description = "Import conversation from JSON"
    category = "Export/Import"
    usage = "/import <filepath>"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if not args:
            context.renderer.error("Usage: /import <filepath>")
            return True
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        import json
        from pathlib import Path

        path = Path(args)
        if not path.exists():
            context.renderer.error(f"File not found: {path}")
            return True
        try:
            data = json.loads(path.read_text("utf-8"))
            for item in data:
                from bytecli.memory.types import MessageEntry

                await context.memory.add_message(
                    MessageEntry(
                        role=item.get("role", "user"),
                        content=item.get("content", ""),
                        token_count=item.get("token_count", 0),
                    )
                )
            context.renderer.info(f"Imported {len(data)} messages from {path}")
        except (json.JSONDecodeError, KeyError) as e:
            context.renderer.error(f"Failed to import: {e}")
        return True
