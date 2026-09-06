from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from bytecli.cli.rendering import RichRenderer
from bytecli.core.agent import AgentLoop
from bytecli.memory.manager import MemoryManager


@dataclass
class CommandContext:
    agent: AgentLoop
    renderer: RichRenderer
    memory: MemoryManager | None = None
    session: Any | None = None
    plugin_manager: Any | None = None
    skill_manager: Any | None = None


class Command(ABC):
    name: str = ""
    aliases: list[str] = []  # noqa: RUF012
    description: str = ""
    category: str = "General"
    usage: str = ""

    @abstractmethod
    async def execute(self, args: str, context: CommandContext) -> bool: ...
