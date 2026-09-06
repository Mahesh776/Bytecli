from abc import ABC
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bytecli.core.events import EventBus


@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    path: Path | None = None
    enabled: bool = True
    hooks: int = 0


@dataclass
class PluginContext:
    event_bus: EventBus | None = None
    data_dir: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)


HookHandler = Callable[..., Awaitable[None]]


class Plugin(ABC):
    name: str = ""
    version: str = "1.0.0"
    description: str = ""

    async def on_load(self, context: PluginContext) -> None:
        pass

    async def on_unload(self) -> None:
        pass

    def get_hooks(self) -> dict[str, HookHandler]:
        return {}
