from pathlib import Path
from typing import Any

from loguru import logger

from bytecli.core.events import EventBus
from bytecli.plugins.base import Plugin, PluginContext, PluginMetadata
from bytecli.plugins.hooks import HookRegistry
from bytecli.plugins.loader import PluginLoader
from bytecli.plugins.registry import PluginRegistry


class PluginManager:
    def __init__(
        self,
        plugin_dirs: list[Path] | None = None,
        event_bus: EventBus | None = None,
        safe_mode: bool = True,
        enabled_plugins: list[str] | None = None,
    ) -> None:
        self._loader = PluginLoader(plugin_dirs or [])
        self._event_bus = event_bus
        self._hook_registry = HookRegistry()
        self._safe_mode = safe_mode
        self._enabled_plugins = enabled_plugins
        self._initialized = False

    @property
    def hook_registry(self) -> HookRegistry:
        return self._hook_registry

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> list[str]:
        if self._initialized:
            return []
        self._initialized = True
        loaded: list[str] = []
        discovered = self._loader.discover()
        for name, path in discovered:
            try:
                await self._load_and_register(name, path)
                loaded.append(name)
            except Exception:
                if self._safe_mode:
                    logger.exception("Failed to load plugin '{}' from {}", name, path)
                else:
                    raise
        if self._event_bus is not None:
            await self._event_bus.publish(
                self._event_bus.create_event("plugin.load", {"plugins": loaded})
            )
        return loaded

    async def load(self, path: Path) -> Plugin:
        plugin = self._loader.load_plugin(path)
        await self._register_plugin(plugin, path)
        if self._event_bus is not None:
            await self._event_bus.publish(
                self._event_bus.create_event("plugin.load", {"plugin": plugin.name})
            )
        return plugin

    async def load_from_source(self, name: str, source_code: str) -> Plugin:
        plugin = self._loader.load_plugin_from_source(name, source_code)
        await self._register_plugin(plugin, source_path=f"<source:{name}>")
        if self._event_bus is not None:
            await self._event_bus.publish(
                self._event_bus.create_event("plugin.load", {"plugin": plugin.name})
            )
        return plugin

    async def _load_and_register(self, name: str, path: Path) -> None:
        plugin = self._loader.load_plugin(path)
        await self._register_plugin(plugin, path)

    async def _register_plugin(self, plugin: Plugin, path: Path | None = None, source_path: str | None = None) -> None:
        meta = PluginMetadata(
            name=plugin.name,
            version=plugin.version,
            description=plugin.description,
            path=path,
        )
        PluginRegistry.register(plugin, meta)

        context = PluginContext(
            event_bus=self._event_bus,
            data_dir=path.parent if path else None,
        )
        await plugin.on_load(context)

        hook_count = 0
        for hook_name, handler in plugin.get_hooks().items():
            self._hook_registry.register(hook_name, handler, plugin.name)
            hook_count += 1
        meta.hooks = hook_count

    async def unload(self, name: str) -> bool:
        plugin = PluginRegistry.get(name)
        if plugin is None:
            return False
        self._hook_registry.unregister_all(name)
        await plugin.on_unload()
        PluginRegistry.unregister(name)
        if self._event_bus is not None:
            await self._event_bus.publish(
                self._event_bus.create_event("plugin.unload", {"plugin": name})
            )
        return True

    async def fire_hook(self, hook: str, **kwargs: Any) -> None:
        if not self._initialized:
            return
        try:
            await self._hook_registry.execute(hook, **kwargs)
        except Exception:
            if not self._safe_mode:
                raise

    def list_plugins(self) -> list[PluginMetadata]:
        return PluginRegistry.list_metadata()

    async def shutdown(self) -> None:
        for plugin in PluginRegistry.list_plugins():
            try:
                await plugin.on_unload()
            except Exception:
                if not self._safe_mode:
                    raise
        PluginRegistry.clear()
        self._hook_registry.clear()
        self._initialized = False
