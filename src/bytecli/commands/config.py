
from bytecli.commands.base import Command, CommandContext
from bytecli.config.schema import ProviderType


class ModelCommand(Command):
    name = "model"
    description = "Show or set the current model"
    category = "Configuration"
    usage = "/model [name]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args:
            context.agent.config.model = args
            context.renderer.info(f"Model set to: {args}")
        else:
            context.renderer.info(f"Current model: {context.agent.config.model}")
        return True


class ProviderCommand(Command):
    name = "provider"
    description = "Show or switch the LLM provider"
    category = "Configuration"
    usage = "/provider [name]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args:
            try:
                pt = ProviderType(args)
                if context.session is not None:
                    context.renderer.info(f"Provider would switch to: {pt.value} (requires restart)")
            except ValueError:
                valid = ", ".join(p.value for p in ProviderType)
                context.renderer.error(f"Invalid provider. Valid options: {valid}")
        else:
            context.renderer.info("Provider switching requires config changes and restart.")
        return True


class TemperatureCommand(Command):
    name = "temperature"
    aliases = ["temp"]  # noqa: RUF012
    description = "Show or set the model temperature (0.0-2.0)"
    category = "Configuration"
    usage = "/temperature [value]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args:
            try:
                val = float(args)
                if val < 0.0 or val > 2.0:
                    context.renderer.error("Temperature must be between 0.0 and 2.0")
                    return True
                context.agent.config.temperature = val
                context.renderer.info(f"Temperature set to: {val}")
            except ValueError:
                context.renderer.error("Temperature must be a number.")
        else:
            context.renderer.info(f"Current temperature: {context.agent.config.temperature}")
        return True


class SystemCommand(Command):
    name = "system"
    description = "Set or show the system prompt"
    category = "Configuration"
    usage = "/system [prompt]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args:
            context.agent.config.system_prompt = args
            context.renderer.info(f"System prompt set ({len(args)} chars).")
            return True
        effective = context.agent.effective_system_prompt()
        source = "custom" if context.agent.config.system_prompt else "auto (based on model size)"
        context.renderer.info(f"Current system prompt [{source}]:\n{effective}")
        return True


class MaxTokensCommand(Command):
    name = "max_tokens"
    aliases = ["maxtokens"]  # noqa: RUF012
    description = "Show or set max tokens per response"
    category = "Configuration"
    usage = "/max_tokens [count]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args:
            try:
                val = int(args)
                if val < 1 or val > 131072:
                    context.renderer.error("Max tokens must be between 1 and 131072")
                    return True
                context.agent.config.max_tokens = val
                context.renderer.info(f"Max tokens set to: {val}")
            except ValueError:
                context.renderer.error("Max tokens must be an integer.")
        else:
            context.renderer.info(f"Current max tokens: {context.agent.config.max_tokens}")
        return True


class TopPCommand(Command):
    name = "top_p"
    aliases = ["topp"]  # noqa: RUF012
    description = "Show or set top_p sampling parameter"
    category = "Configuration"
    usage = "/top_p [value]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args:
            try:
                val = float(args)
                if val < 0.0 or val > 1.0:
                    context.renderer.error("top_p must be between 0.0 and 1.0")
                    return True
                context.agent.config.config.top_p = val  # type: ignore[attr-defined]
                context.renderer.info(f"top_p set to: {val}")
            except ValueError:
                context.renderer.error("top_p must be a number.")
        else:
            context.renderer.info("top_p must be set via config.")
        return True


class ConfigCommand(Command):
    name = "config"
    description = "Show current configuration summary"
    category = "Configuration"
    usage = "/config [key] [value]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        c = context.agent.config
        lines = [
            f"Model: {c.model}",
            f"Temperature: {c.temperature}",
            f"Max tokens: {c.max_tokens}",
            f"System prompt: {'custom' if c.system_prompt else 'auto'}",
            f"Max turns: {c.max_turns}",
        ]
        context.renderer.info("Configuration:\n" + "\n".join(f"  {line}" for line in lines))
        return True
