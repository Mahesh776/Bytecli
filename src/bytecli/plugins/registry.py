from bytecli.plugins.base import Plugin, PluginMetadata


class PluginRegistry:
    _plugins: dict[str, Plugin] = {}  # noqa: RUF012
    _metadatas: dict[str, PluginMetadata] = {}  # noqa: RUF012

    @classmethod
    def register(cls, plugin: Plugin, metadata: PluginMetadata | None = None) -> None:
        cls._plugins[plugin.name] = plugin
        if metadata is not None:
            cls._metadatas[plugin.name] = metadata
        else:
            cls._metadatas[plugin.name] = PluginMetadata(
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
            )

    @classmethod
    def get(cls, name: str) -> Plugin | None:
        return cls._plugins.get(name)

    @classmethod
    def get_metadata(cls, name: str) -> PluginMetadata | None:
        return cls._metadatas.get(name)

    @classmethod
    def list_plugins(cls) -> list[Plugin]:
        return list(cls._plugins.values())

    @classmethod
    def list_metadata(cls) -> list[PluginMetadata]:
        return list(cls._metadatas.values())

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._plugins.pop(name, None)
        cls._metadatas.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        cls._plugins.clear()
        cls._metadatas.clear()

    @classmethod
    def is_loaded(cls, name: str) -> bool:
        return name in cls._plugins
