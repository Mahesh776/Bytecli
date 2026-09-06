import tempfile
from pathlib import Path

import pytest

from bytecli.core.errors import PluginLoadError
from bytecli.core.events import EventBus
from bytecli.plugins.base import Plugin, PluginContext, PluginMetadata
from bytecli.plugins.hooks import (
    ALL_HOOKS,
    HOOK_AGENT_BEFORE_ACT,
    HOOK_AGENT_BEFORE_THINK,
    HOOK_AGENT_BEFORE_TURN,
    HOOK_AGENT_AFTER_TURN,
    HOOK_TOOL_BEFORE_EXECUTE,
    HOOK_TOOL_AFTER_EXECUTE,
    HookRegistry,
)
from bytecli.plugins.loader import PluginLoader
from bytecli.plugins.manager import PluginManager
from bytecli.plugins.registry import PluginRegistry


class TestPluginBase:
    def test_plugin_defaults(self) -> None:
        class SimplePlugin(Plugin):
            name = "test"
            description = "A test plugin"

        p = SimplePlugin()
        assert p.name == "test"
        assert p.description == "A test plugin"
        assert p.version == "1.0.0"

    async def test_plugin_lifecycle(self) -> None:
        events: list[str] = []

        class LifecyclePlugin(Plugin):
            name = "lifecycle"

            async def on_load(self, context: PluginContext) -> None:
                events.append("load")

            async def on_unload(self) -> None:
                events.append("unload")

        p = LifecyclePlugin()
        await p.on_load(PluginContext())
        await p.on_unload()
        assert events == ["load", "unload"]

    async def test_plugin_get_hooks_default(self) -> None:
        class NoHooksPlugin(Plugin):
            name = "nohooks"

        p = NoHooksPlugin()
        assert p.get_hooks() == {}

    async def test_plugin_get_hooks_custom(self) -> None:
        class HookedPlugin(Plugin):
            name = "hooked"

            async def my_hook(self, **kwargs: object) -> None:
                pass

            def get_hooks(self) -> dict[str, object]:
                return {"agent.before_turn": self.my_hook}

        p = HookedPlugin()
        hooks = p.get_hooks()
        assert "agent.before_turn" in hooks


class TestPluginMetadata:
    def test_defaults(self) -> None:
        meta = PluginMetadata(name="p1", version="2.0.0", description="desc")
        assert meta.name == "p1"
        assert meta.version == "2.0.0"
        assert meta.description == "desc"
        assert meta.path is None
        assert meta.enabled is True
        assert meta.hooks == 0

    def test_full(self) -> None:
        meta = PluginMetadata(
            name="p1",
            version="1.0.0",
            description="desc",
            path=Path("/tmp/test.py"),
            enabled=False,
            hooks=3,
        )
        assert meta.path == Path("/tmp/test.py")
        assert meta.enabled is False
        assert meta.hooks == 3


class TestHookRegistry:
    def test_register_and_execute(self) -> None:
        registry = HookRegistry()
        results: list[str] = []

        async def handler(**kwargs: object) -> None:
            results.append(kwargs.get("msg", ""))

        registry.register("test.hook", handler, "plugin_a")
        assert "test.hook" in registry._handlers

    async def test_execute_calls_handlers(self) -> None:
        registry = HookRegistry()
        results: list[int] = []

        async def handler_a(**kwargs: object) -> None:
            results.append(1)

        async def handler_b(**kwargs: object) -> None:
            results.append(2)

        registry.register("test.hook", handler_a, "a", priority=10)
        registry.register("test.hook", handler_b, "b", priority=20)
        await registry.execute("test.hook")
        assert results == [1, 2]

    async def test_execute_priorities(self) -> None:
        registry = HookRegistry()
        results: list[int] = []

        async def handler_high(**kwargs: object) -> None:
            results.append(1)

        async def handler_low(**kwargs: object) -> None:
            results.append(2)

        registry.register("test.hook", handler_low, "low", priority=50)
        registry.register("test.hook", handler_high, "high", priority=10)
        await registry.execute("test.hook")
        assert results == [1, 2]

    async def test_execute_no_handlers(self) -> None:
        registry = HookRegistry()
        await registry.execute("nonexistent.hook")

    def test_unregister_all(self) -> None:
        registry = HookRegistry()

        async def handler(**kwargs: object) -> None:
            pass

        registry.register("a.hook", handler, "plugin_x")
        registry.register("b.hook", handler, "plugin_x")
        registry.register("a.hook", handler, "plugin_y")
        assert len(registry.handlers_for_hook("a.hook")) == 2
        assert len(registry.handlers_for_hook("b.hook")) == 1
        registry.unregister_all("plugin_x")
        assert len(registry.handlers_for_hook("a.hook")) == 1
        assert "b.hook" not in registry._handlers

    def test_list_hooks(self) -> None:
        registry = HookRegistry()

        async def handler(**kwargs: object) -> None:
            pass

        registry.register("hook.a", handler, "p1")
        registry.register("hook.b", handler, "p1")
        hooks = registry.list_hooks()
        assert "hook.a" in hooks
        assert "hook.b" in hooks

    def test_clear(self) -> None:
        registry = HookRegistry()

        async def handler(**kwargs: object) -> None:
            pass

        registry.register("hook.a", handler, "p1")
        registry.clear()
        assert registry.list_hooks() == []

    def test_handlers_for_hook(self) -> None:
        registry = HookRegistry()

        async def handler(**kwargs: object) -> None:
            pass

        registry.register("hook.a", handler, "p1", priority=10)
        result = registry.handlers_for_hook("hook.a")
        assert len(result) == 1
        assert result[0] == (10, "p1")

    def test_handlers_for_hook_missing(self) -> None:
        registry = HookRegistry()
        assert registry.handlers_for_hook("missing") == []


class TestAllHooks:
    def test_all_hooks_count(self) -> None:
        assert len(ALL_HOOKS) == 29

    def test_specific_hooks_present(self) -> None:
        essential = [
            HOOK_AGENT_BEFORE_TURN,
            HOOK_AGENT_AFTER_TURN,
            HOOK_AGENT_BEFORE_THINK,
            HOOK_AGENT_BEFORE_ACT,
            HOOK_TOOL_BEFORE_EXECUTE,
            HOOK_TOOL_AFTER_EXECUTE,
        ]
        for h in essential:
            assert h in ALL_HOOKS

    def test_all_hooks_unique(self) -> None:
        assert len(ALL_HOOKS) == len(set(ALL_HOOKS))


class TestPluginRegistry:
    def setup_method(self) -> None:
        PluginRegistry.clear()

    def test_register_and_get(self) -> None:
        class TestPlugin(Plugin):
            name = "test_plugin"

        plugin = TestPlugin()
        PluginRegistry.register(plugin)
        assert PluginRegistry.get("test_plugin") is plugin

    def test_register_with_metadata(self) -> None:
        class TestPlugin(Plugin):
            name = "meta_plugin"
            version = "2.0.0"
            description = "meta desc"

        plugin = TestPlugin()
        meta = PluginMetadata(name="meta_plugin", version="2.0.0", description="meta desc")
        PluginRegistry.register(plugin, meta)
        retrieved = PluginRegistry.get_metadata("meta_plugin")
        assert retrieved is not None
        assert retrieved.version == "2.0.0"

    def test_get_nonexistent(self) -> None:
        assert PluginRegistry.get("nonexistent") is None

    def test_get_metadata_nonexistent(self) -> None:
        assert PluginRegistry.get_metadata("nonexistent") is None

    def test_list_plugins(self) -> None:
        class A(Plugin):
            name = "a"
        class B(Plugin):
            name = "b"

        PluginRegistry.register(A())
        PluginRegistry.register(B())
        plugins = PluginRegistry.list_plugins()
        assert len(plugins) == 2
        names = {p.name for p in plugins}
        assert names == {"a", "b"}

    def test_list_metadata(self) -> None:
        class A(Plugin):
            name = "a"
        PluginRegistry.register(A(), PluginMetadata(name="a", version="1.0.0", description=""))
        metas = PluginRegistry.list_metadata()
        assert len(metas) >= 1

    def test_unregister(self) -> None:
        class A(Plugin):
            name = "temp"
        PluginRegistry.register(A())
        assert PluginRegistry.is_loaded("temp")
        PluginRegistry.unregister("temp")
        assert not PluginRegistry.is_loaded("temp")

    def test_is_loaded(self) -> None:
        class A(Plugin):
            name = "loaded_plugin"
        assert not PluginRegistry.is_loaded("loaded_plugin")
        PluginRegistry.register(A())
        assert PluginRegistry.is_loaded("loaded_plugin")

    def test_clear(self) -> None:
        class A(Plugin):
            name = "p1"
        class B(Plugin):
            name = "p2"
        PluginRegistry.register(A())
        PluginRegistry.register(B())
        PluginRegistry.clear()
        assert PluginRegistry.list_plugins() == []


class TestPluginLoader:
    def test_discover_no_dirs(self) -> None:
        loader = PluginLoader()
        assert loader.discover() == []

    def test_discover_missing_dir(self) -> None:
        loader = PluginLoader([Path("/nonexistent/path")])
        assert loader.discover() == []

    def test_discover_finds_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "my_plugin.py").write_text("""
from bytecli.plugins.base import Plugin

class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0.0"
    description = "My test plugin"
""")
            (d / "_private.py").write_text("private")
            (d / "not_python.txt").write_text("not a plugin")
            loader = PluginLoader([d])
            found = loader.discover()
            names = [n for n, _ in found]
            assert "my_plugin" in names
            assert "_private" not in names

    def test_load_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            plugin_file = d / "test_plugin.py"
            plugin_file.write_text("""
from bytecli.plugins.base import Plugin

class TestPlugin(Plugin):
    name = "test_plugin"
    version = "0.1.0"
    description = "A test plugin"
""")
            loader = PluginLoader()
            plugin = loader.load_plugin(plugin_file)
            assert plugin.name == "test_plugin"
            assert plugin.version == "0.1.0"
            assert plugin.description == "A test plugin"

    def test_load_plugin_no_plugin_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            plugin_file = d / "bad_plugin.py"
            plugin_file.write_text("x = 1")
            loader = PluginLoader()
            with pytest.raises(ValueError, match="No Plugin subclass"):
                loader.load_plugin(plugin_file)

    def test_load_plugin_from_source(self) -> None:
        source = """
from bytecli.plugins.base import Plugin

class InlinePlugin(Plugin):
    name = "inline_plugin"
    version = "3.0.0"
"""
        loader = PluginLoader()
        plugin = loader.load_plugin_from_source("inline", source)
        assert plugin.name == "inline_plugin"
        assert plugin.version == "3.0.0"


class TestPluginManager:
    def setup_method(self) -> None:
        PluginRegistry.clear()

    async def test_initialize_no_dirs(self) -> None:
        mgr = PluginManager(plugin_dirs=[])
        loaded = await mgr.initialize()
        assert loaded == []
        assert mgr.is_initialized

    async def test_initialize_discover_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "alpha.py").write_text("""
from bytecli.plugins.base import Plugin

class AlphaPlugin(Plugin):
    name = "alpha"
    version = "1.0.0"
    description = "Alpha plugin"
""")
            mgr = PluginManager(plugin_dirs=[d])
            loaded = await mgr.initialize()
            assert "alpha" in loaded
            assert PluginRegistry.is_loaded("alpha")

    async def test_initialize_safe_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "broken.py").write_text("this is not valid python ^^^")
            mgr = PluginManager(plugin_dirs=[d], safe_mode=True)
            loaded = await mgr.initialize()
            assert loaded == []

    async def test_load_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            plugin_file = d / "load_test.py"
            plugin_file.write_text("""
from bytecli.plugins.base import Plugin

class LoadTestPlugin(Plugin):
    name = "load_test"
""")
            mgr = PluginManager()
            plugin = await mgr.load(plugin_file)
            assert plugin.name == "load_test"
            assert PluginRegistry.is_loaded("load_test")

    async def test_load_from_source(self) -> None:
        source = """
from bytecli.plugins.base import Plugin

class SourcePlugin(Plugin):
    name = "source_plugin"
"""
        mgr = PluginManager()
        plugin = await mgr.load_from_source("source_test", source)
        assert plugin.name == "source_plugin"
        assert PluginRegistry.is_loaded("source_plugin")

    async def test_unload(self) -> None:
        source = """
from bytecli.plugins.base import Plugin

class UnloadPlugin(Plugin):
    name = "unload_plugin"
"""
        mgr = PluginManager()
        await mgr.load_from_source("unload_test", source)
        assert PluginRegistry.is_loaded("unload_plugin")
        result = await mgr.unload("unload_plugin")
        assert result is True
        assert not PluginRegistry.is_loaded("unload_plugin")

    async def test_unload_nonexistent(self) -> None:
        mgr = PluginManager()
        result = await mgr.unload("nonexistent")
        assert result is False

    async def test_fire_hook(self) -> None:
        results: list[str] = []

        class HookPlugin(Plugin):
            name = "hook_plugin"

            async def on_turn_start(self, **kwargs: object) -> None:
                results.append(f"turn_{kwargs.get('turn', '?')}")

            def get_hooks(self) -> dict[str, object]:
                return {"agent.before_turn": self.on_turn_start}

        source = """
from bytecli.plugins.base import Plugin

class FireHookPlugin(Plugin):
    name = "fire_hook_plugin"

    async def on_turn_start(self, **kwargs):
        import sys
        sys.stdout.write(f"turn_{kwargs.get('turn', '?')}")

    def get_hooks(self):
        return {"agent.before_turn": self.on_turn_start}
"""
        mgr = PluginManager()
        await mgr.load_from_source("fire_hook_test", source)
        await mgr.fire_hook("agent.before_turn", turn=1)
        assert mgr.hook_registry.list_hooks() == ["agent.before_turn"]

    async def test_hook_execution_order(self) -> None:
        results: list[str] = []

        async def handler_first(**kwargs: object) -> None:
            results.append("first")

        async def handler_second(**kwargs: object) -> None:
            results.append("second")

        mgr = PluginManager()
        await mgr.initialize()
        mgr._hook_registry.register("test.order", handler_first, "a", priority=5)
        mgr._hook_registry.register("test.order", handler_second, "b", priority=10)
        await mgr.fire_hook("test.order")
        assert results == ["first", "second"]

    async def test_fire_hook_before_init(self) -> None:
        mgr = PluginManager()
        assert mgr.is_initialized is False
        await mgr.fire_hook("agent.before_turn")
        assert True

    async def test_list_plugins(self) -> None:
        source = """
from bytecli.plugins.base import Plugin

class ListPlugin(Plugin):
    name = "list_plugin"
    version = "2.0.0"
    description = "list me"
"""
        mgr = PluginManager()
        await mgr.load_from_source("list_test", source)
        plugins = mgr.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "list_plugin"
        assert plugins[0].version == "2.0.0"
        assert plugins[0].description == "list me"

    async def test_shutdown(self) -> None:
        unloaded: list[str] = []

        class ShutdownPlugin(Plugin):
            name = "shutdown_test"

            async def on_unload(self) -> None:
                unloaded.append(self.name)

        source = """
from bytecli.plugins.base import Plugin

class ShutdownPlugin2(Plugin):
    name = "shutdown_test2"

    async def on_unload(self):
        import sys
        sys.stdout.write("unloaded")
"""
        mgr = PluginManager()
        await mgr.load_from_source("shutdown1", """
from bytecli.plugins.base import Plugin

class ShutdownPlugin1(Plugin):
    name = "shutdown_plugin1"
""")
        await mgr.load_from_source("shutdown2", """
from bytecli.plugins.base import Plugin

class ShutdownPlugin2(Plugin):
    name = "shutdown_plugin2"
""")
        assert len(PluginRegistry.list_plugins()) >= 2
        await mgr.shutdown()
        assert PluginRegistry.list_plugins() == []
        assert not mgr.is_initialized

    async def test_double_initialize(self) -> None:
        mgr = PluginManager()
        await mgr.initialize()
        result = await mgr.initialize()
        assert result == []

    async def test_event_bus_integration(self) -> None:
        events: list[str] = []
        bus = EventBus(auto_log=False)

        async def collector(event: object) -> None:
            events.append(str(event))

        source_a = """
from bytecli.plugins.base import Plugin

class EventPluginA(Plugin):
    name = "event_plugin_a"
"""
        source_b = """
from bytecli.plugins.base import Plugin

class EventPluginB(Plugin):
    name = "event_plugin_b"
"""
        mgr = PluginManager(event_bus=bus)
        await mgr.load_from_source("evt_a", source_a)
        await mgr.load_from_source("evt_b", source_b)
        assert PluginRegistry.is_loaded("event_plugin_a")
        assert PluginRegistry.is_loaded("event_plugin_b")


class TestHookConstants:
    def test_agent_hooks_prefix(self) -> None:
        for hook in ALL_HOOKS:
            if hook.startswith("agent."):
                assert "." in hook

    def test_tool_hooks_prefix(self) -> None:
        tool_hooks = [h for h in ALL_HOOKS if h.startswith("tool.")]
        assert len(tool_hooks) >= 3

    def test_session_hooks_present(self) -> None:
        session_hooks = [h for h in ALL_HOOKS if h.startswith("session.")]
        assert len(session_hooks) >= 4

    def test_memory_hooks_present(self) -> None:
        memory_hooks = [h for h in ALL_HOOKS if h.startswith("memory.")]
        assert len(memory_hooks) >= 5

    def test_config_hooks_present(self) -> None:
        config_hooks = [h for h in ALL_HOOKS if h.startswith("config.")]
        assert len(config_hooks) >= 3
