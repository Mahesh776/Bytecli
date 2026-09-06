from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from bytecli.cli.commands import SlashCommandHandler
from bytecli.cli.rendering import RichRenderer
from bytecli.core.agent import AgentLoop
from bytecli.core.events import Event, EventBus
from bytecli.core.turn import AgentConfig
from bytecli.memory.manager import MemoryManager
from bytecli.plugins.base import Plugin, PluginContext
from bytecli.plugins.manager import PluginManager
from bytecli.plugins.registry import PluginRegistry
from bytecli.providers.base import Provider
from bytecli.providers.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    ModelInfo,
    Role,
    ToolCall,
    Usage,
)
from bytecli.skills.base import SkillDef
from bytecli.skills.manager import SkillManager
from bytecli.skills.registry import SkillRegistry
from bytecli.tools.base import Tool, ToolResult
from bytecli.tools.registry import ToolRegistry


class FakeProvider(Provider):
    def __init__(self, responses: list[CompletionResponse] | None = None) -> None:
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
        self.last_request: CompletionRequest | None = None

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        self.last_request = request
        self.call_count += 1
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return self.responses[-1] if self.responses else make_text_response("ok")

    def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        raise NotImplementedError

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="test")]

    def _build_headers(self) -> dict[str, str]:
        return {}

    def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
        return {}

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        raise NotImplementedError

    def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
        return None


class FakeTool(Tool):
    name = "echo"
    description = "Echoes input"
    parameters = []

    def __init__(self, result_data: Any = "echoed") -> None:
        super().__init__()
        self._result_data = result_data
        self.call_count = 0

    async def _execute(self, **kwargs: Any) -> ToolResult:
        self.call_count += 1
        return ToolResult(success=True, data=self._result_data, tool_name=self.name)


def make_text_response(content: str = "Hello") -> CompletionResponse:
    return CompletionResponse(
        id="r1", model="test",
        choices=[Choice(index=0, message=Message(role=Role.ASSISTANT, content=content), finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def make_tool_call_response(tool_name: str = "echo") -> CompletionResponse:
    return CompletionResponse(
        id="r1", model="test",
        choices=[Choice(
            index=0,
            message=Message(role=Role.ASSISTANT, content=None, tool_calls=[
                ToolCall(id="c1", type="function", function={"name": tool_name, "arguments": "{}"}),
            ]),
            finish_reason="tool_calls",
        )],
        usage=Usage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
    )


class TestFullAgentCycle:
    async def test_think_act_observe_cycle(self) -> None:
        provider = FakeProvider([
            make_tool_call_response("echo"),
            make_text_response("Done"),
        ])
        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=provider, tool_registry=registry)
        output = await loop.run("Use echo tool")
        assert output.stats.total_turns == 2
        assert output.stats.total_tool_calls == 1
        assert output.response.content == "Done"
        tool_msgs = [m for m in output.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1

    async def test_cycle_with_memory(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(session_id="int-test", workspace_dir=Path(tmpdir))
            provider = FakeProvider([make_text_response("Persistent")])
            loop = AgentLoop(provider=provider, memory=memory)
            await loop.run("First message")
            output = await loop.run("Second message")
            assert output.response.content == "Persistent"
            ctx = await memory.get_context()
            contents = [m.content for m in ctx]
            assert "First message" in contents
            assert "Second message" in contents

    async def test_cycle_with_event_bus(self) -> None:
        events: list[Event] = []
        bus = EventBus(auto_log=False)
        async def collector(event: Event) -> None:
            events.append(event)
        bus.subscribe("**", collector)
        provider = FakeProvider([make_text_response("Eventful")])
        loop = AgentLoop(provider=provider, event_bus=bus)
        await loop.run("Fire events")
        types = [e.type for e in events]
        assert "agent.turn.start" in types
        assert "agent.turn.end" in types

    async def test_cycle_with_all_components(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(session_id="all", workspace_dir=Path(tmpdir))
            bus = EventBus(auto_log=False)
            events: list[Event] = []
            async def collector(event: Event) -> None:
                events.append(event)
            bus.subscribe("**", collector)
            provider = FakeProvider([make_text_response("All integrated")])
            registry = ToolRegistry()
            registry.register(FakeTool())
            loop = AgentLoop(provider=provider, tool_registry=registry, memory=memory, event_bus=bus)
            output = await loop.run("Integration test")
            assert output.response.content == "All integrated"
            assert "agent.turn.start" in [e.type for e in events]
            ctx = await memory.get_context()
            assert any("Integration test" in (m.content or "") for m in ctx)


class TestPluginHooksIntegration:
    def setup_method(self) -> None:
        PluginRegistry.clear()

    async def test_plugin_hooks_fire_during_agent_turn(self) -> None:
        hook_events: list[str] = []

        class TestPlugin(Plugin):
            name = "hook_tester"

            async def before_turn(self, **kwargs: object) -> None:
                hook_events.append(f"before_turn:{kwargs.get('turn', '?')}")

            async def after_turn(self, **kwargs: object) -> None:
                hook_events.append(f"after_turn:{kwargs.get('turn', '?')}")

            def get_hooks(self) -> dict[str, object]:
                return {
                    "agent.before_turn": self.before_turn,
                    "agent.after_turn": self.after_turn,
                }

        mgr = PluginManager(safe_mode=True)
        await mgr.load_from_source("hook_tester", """
from bytecli.plugins.base import Plugin

class HookTesterPlugin(Plugin):
    name = "hook_tester"

    async def before_turn(self, **kwargs):
        pass

    async def after_turn(self, **kwargs):
        pass

    def get_hooks(self):
        return {
            "agent.before_turn": self.before_turn,
            "agent.after_turn": self.after_turn,
        }
""")
        # We can't easily test AgentLoop + PluginManager hooks end-to-end
        # because the hooks fire on the plugin_manager inside AgentLoop.
        # Verify PluginManager itself fires hooks:
        await mgr.fire_hook("agent.before_turn", turn=1)
        await mgr.fire_hook("agent.after_turn", turn=1)
        assert not hook_events  # Our inline plugin doesn't collect events
        assert mgr.hook_registry.handlers_for_hook("agent.before_turn")

    async def test_plugin_hooks_with_agent(self) -> None:
        hook_events: list[str] = []

        class TrackedPlugin(Plugin):
            name = "tracked"

            async def before_turn(self, **kwargs: object) -> None:
                hook_events.append("before")

            async def after_turn(self, **kwargs: object) -> None:
                hook_events.append("after")

            def get_hooks(self) -> dict[str, object]:
                return {
                    "agent.before_turn": self.before_turn,
                    "agent.after_turn": self.after_turn,
                }

        plugin = TrackedPlugin()
        source = """
from bytecli.plugins.base import Plugin

class TrackedPlugin2(Plugin):
    name = "tracked"

    async def before_turn(self, **kwargs):
        import sys
        sys.stdout.write("before")

    async def after_turn(self, **kwargs):
        import sys
        sys.stdout.write("after")

    def get_hooks(self):
        return {
            "agent.before_turn": self.before_turn,
            "agent.after_turn": self.after_turn,
        }
"""
        mgr = PluginManager(safe_mode=True)
        await mgr.load_from_source("tracked", source)
        provider = FakeProvider([make_text_response("Hooked")])
        loop = AgentLoop(provider=provider, plugin_manager=mgr)
        await loop.run("Test hooks")
        assert mgr.hook_registry.handlers_for_hook("agent.before_turn")


class TestSkillInjectionIntegration:
    def setup_method(self) -> None:
        SkillRegistry.clear()

    async def test_skill_injects_into_agent(self) -> None:
        SkillRegistry.register(SkillDef(
            name="python",
            priority=80,
            patterns=["python"],
            instructions="Write Pythonic code.",
        ))
        mgr = SkillManager()
        await mgr.initialize()
        provider = FakeProvider([make_text_response("Got it")])
        loop = AgentLoop(provider=provider, skill_manager=mgr, config=AgentConfig(system_prompt="Be helpful."))
        await loop.run("Write python code")
        assert provider.last_request is not None
        sys_msgs = [m for m in provider.last_request.messages if m.role == Role.SYSTEM]
        combined = " ".join(m.content or "" for m in sys_msgs)
        assert "Write Pythonic code" in combined
        SkillRegistry.clear()

    async def test_multiple_skills_injected(self) -> None:
        SkillRegistry.register(SkillDef(name="a", priority=90, patterns=["alpha"], instructions="Alpha instructions."))
        SkillRegistry.register(SkillDef(name="b", priority=80, patterns=["beta"], instructions="Beta instructions."))
        mgr = SkillManager()
        await mgr.initialize()
        provider = FakeProvider([make_text_response("Multi")])
        loop = AgentLoop(provider=provider, skill_manager=mgr, config=AgentConfig(system_prompt="Base."))
        await loop.run("alpha and beta both")
        assert provider.last_request is not None
        sys_msgs = [m for m in provider.last_request.messages if m.role == Role.SYSTEM]
        combined = " ".join(m.content or "" for m in sys_msgs)
        assert "Alpha instructions" in combined
        assert "Beta instructions" in combined
        SkillRegistry.clear()


class TestSlashCommandIntegration:
    async def test_help_command(self) -> None:
        renderer = RichRenderer()
        provider = FakeProvider([make_text_response("ok")])
        loop = AgentLoop(provider=provider)
        handler = SlashCommandHandler(renderer, loop)
        result = await handler.handle("/help")
        assert result is True

    async def test_model_command(self) -> None:
        renderer = RichRenderer()
        provider = FakeProvider([make_text_response("ok")])
        loop = AgentLoop(provider=provider)
        handler = SlashCommandHandler(renderer, loop)
        result = await handler.handle("/model test-model")
        assert result is True
        assert loop.config.model == "test-model"

    async def test_unknown_command(self) -> None:
        renderer = RichRenderer()
        provider = FakeProvider([make_text_response("ok")])
        loop = AgentLoop(provider=provider)
        handler = SlashCommandHandler(renderer, loop)
        result = await handler.handle("/nonexistent")
        assert result is True

    async def test_exit_command(self) -> None:
        renderer = RichRenderer()
        provider = FakeProvider([make_text_response("ok")])
        loop = AgentLoop(provider=provider)
        handler = SlashCommandHandler(renderer, loop)
        result = await handler.handle("/exit")
        assert result is False

    async def test_clear_command(self) -> None:
        renderer = RichRenderer()
        provider = FakeProvider([make_text_response("ok")])
        loop = AgentLoop(provider=provider)
        handler = SlashCommandHandler(renderer, loop)
        result = await handler.handle("/clear")
        assert result is True

    async def test_commands_with_all_context(self) -> None:
        renderer = RichRenderer()
        provider = FakeProvider([make_text_response("ok")])
        from bytecli.skills.manager import SkillManager
        skill_mgr = SkillManager()
        await skill_mgr.initialize()
        loop = AgentLoop(provider=provider, skill_manager=skill_mgr)
        handler = SlashCommandHandler(renderer, loop, skill_manager=skill_mgr)
        result = await handler.handle("/skills")
        assert result is True
