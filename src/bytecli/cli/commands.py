from bytecli.cli.rendering import RichRenderer
from bytecli.commands.base import CommandContext
from bytecli.commands.registry import CommandRegistry, register_all
from bytecli.core.agent import AgentLoop
from bytecli.memory.manager import MemoryManager


class SlashCommandHandler:
    def __init__(
        self,
        renderer: RichRenderer,
        agent: AgentLoop,
        memory: MemoryManager | None = None,
        session: object | None = None,
        plugin_manager: object | None = None,
        skill_manager: object | None = None,
    ) -> None:
        self._renderer = renderer
        self._agent = agent
        self._memory = memory
        self._session = session
        self._plugin_manager = plugin_manager
        self._skill_manager = skill_manager
        self._registry: CommandRegistry = register_all()

    @property
    def registry(self) -> CommandRegistry:
        return self._registry

    async def handle(self, command: str) -> bool:
        parts = command.strip().split(maxsplit=1)
        cmd_name = parts[0].lstrip("/").lower()
        arg = parts[1] if len(parts) > 1 else ""

        context = CommandContext(
            agent=self._agent,
            renderer=self._renderer,
            memory=self._memory,
            session=self._session,
            plugin_manager=self._plugin_manager,
            skill_manager=self._skill_manager,
        )

        return await self._registry.execute(cmd_name, arg, context)
