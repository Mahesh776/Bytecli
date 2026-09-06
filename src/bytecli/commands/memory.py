from bytecli.commands.base import Command, CommandContext


class MemoryCommand(Command):
    name = "memory"
    description = "Show memory statistics for all stores"
    category = "Memory"
    usage = "/memory"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        stats = await context.memory.memory_stats()
        context.renderer.memory_stats_table(stats)
        return True


class TokensCommand(Command):
    name = "tokens"
    description = "Show conversation token count and message stats"
    category = "Memory"
    usage = "/tokens"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        conv_stats = await context.memory.conversation_stats()
        context.renderer.info(
            f"Messages: {conv_stats.message_count}, Tokens: {conv_stats.total_tokens}, "
            f"Compactions: {conv_stats.compaction_count}"
        )
        return True


class CompactCommand(Command):
    name = "compact"
    description = "Force memory compaction across all stores"
    category = "Memory"
    usage = "/compact"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        result = await context.memory.compact()
        total = sum(result.values())
        context.renderer.info(f"Compaction complete. Removed {total} entries across {len(result)} stores.")
        return True


class ForgetCommand(Command):
    name = "forget"
    description = "Forget a specific memory entry by key"
    category = "Memory"
    usage = "/forget <key>"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if not args:
            context.renderer.error("Usage: /forget <key>")
            return True
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        entry = await context.memory.retrieve_entry(args)
        if entry is None:
            context.renderer.info(f"No memory entry found with key: {args}")
            return True
        await context.memory.session.delete(args)
        if context.memory.project:
            await context.memory.project.delete(args)
        if context.memory.workspace:
            await context.memory.workspace.delete(args)
        context.renderer.info(f"Forgot memory entry: {args}")
        return True


class ContextCommand(Command):
    name = "context"
    description = "Show current conversation context size"
    category = "Memory"
    usage = "/context"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if context.memory is None:
            context.renderer.info("No memory manager available.")
            return True
        conv_stats = await context.memory.conversation_stats()
        total_tokens = conv_stats.total_tokens
        max_tokens = context.agent.config.max_tokens
        pct = (total_tokens / max_tokens * 100) if max_tokens > 0 else 0
        context.renderer.info(
            f"Context: {total_tokens}/{max_tokens} tokens ({pct:.1f}%), "
            f"{conv_stats.message_count} messages"
        )
        return True
