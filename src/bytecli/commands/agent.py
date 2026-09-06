
from bytecli import __version__
from bytecli.commands.base import Command, CommandContext


class StatusCommand(Command):
    name = "status"
    description = "Show agent status and configuration"
    category = "Agent"
    usage = "/status"

    async def execute(self, args: str, context: CommandContext) -> bool:
        c = context.agent.config
        conv_stats = None
        if context.memory is not None:
            conv_stats = await context.memory.conversation_stats()

        lines = [
            f"ByteCli v{__version__}",
            f"Model: {c.model}",
            f"Temperature: {c.temperature}",
            f"Max tokens: {c.max_tokens}",
            f"Max turns: {c.max_turns}",
            f"System prompt: {'custom' if c.system_prompt else 'auto'}",
        ]
        if conv_stats:
            lines.append(f"Messages: {conv_stats.message_count}")
            lines.append(f"Tokens: {conv_stats.total_tokens}")
            lines.append(f"Compactions: {conv_stats.compaction_count}")

        context.renderer.info("Agent Status:\n" + "\n".join(f"  {line}" for line in lines))
        return True


class RetryCommand(Command):
    name = "retry"
    description = "Retry the last agent turn"
    category = "Agent"
    usage = "/retry"

    async def execute(self, args: str, context: CommandContext) -> bool:
        context.renderer.info("Retry: re-run your last input to get a new response.")
        return True


class UndoCommand(Command):
    name = "undo"
    description = "Undo the last conversation turn"
    category = "Agent"
    usage = "/undo"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        msgs = await context.memory.get_context()
        if not msgs:
            context.renderer.info("No messages to undo.")
            return True
        last_user = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                last_user = i
                break
        if last_user is not None:
            await context.memory.conversation.clear()
            remaining = msgs[:last_user]
            for m in remaining:
                from bytecli.memory.types import MessageEntry

                await context.memory.add_message(
                    MessageEntry(role=m.role, content=m.content, token_count=m.token_count)
                )
            context.renderer.info(f"Undone last turn ({len(msgs) - last_user} messages removed).")
        else:
            context.renderer.info("No user message found to undo.")
        return True
