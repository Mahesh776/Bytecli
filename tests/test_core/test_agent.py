import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bytecli.core.agent import AgentLoop
from bytecli.core.errors import AgentLoopError, AgentTerminationError
from bytecli.core.events import Event, EventBus
from bytecli.core.turn import AgentConfig, AgentOutput, TurnRecord, TurnStats, message_entry_to_provider
from bytecli.memory.manager import MemoryManager
from bytecli.memory.types import MessageEntry
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
from bytecli.tools.base import Tool, ToolResult
from bytecli.tools.filesystem import WriteTool
from bytecli.tools.registry import ToolRegistry


class FakeProvider(Provider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.call_count = 0

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        if self.call_count >= len(self.responses):
            return self.responses[-1]
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

    def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        raise NotImplementedError

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="test-model")]

    def _build_headers(self) -> dict[str, str]:
        return {}

    def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
        return {}

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        raise NotImplementedError

    def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
        return None


class FakeTool(Tool):
    name = "test_tool"
    description = "A test tool"
    parameters = []
    name_aliases = {"fake_tool_alt", "the_test_tool"}

    def __init__(self, result_data: Any = "tool_result") -> None:
        super().__init__()
        self._result_data = result_data

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=self._result_data, tool_name=self.name)


class BrokenTool(Tool):
    name = "broken_tool"
    description = "A broken tool"
    parameters = []

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=False, error="Something broke", tool_name=self.name)


def make_text_response(content: str, finish_reason: str = "stop") -> CompletionResponse:
    return CompletionResponse(
        id="resp_1",
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=Message(role=Role.ASSISTANT, content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def make_tool_call_response(
    tool_name: str = "test_tool",
    args: str = '{"key": "value"}',
    tool_call_id: str = "call_1",
) -> CompletionResponse:
    return CompletionResponse(
        id="resp_1",
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=tool_call_id,
                            type="function",
                            function={"name": tool_name, "arguments": args},
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=Usage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
    )


class TestTurnModule:
    def test_agent_config_defaults(self) -> None:
        config = AgentConfig()
        assert config.model == "local-model"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.max_turns == 25
        assert config.system_prompt is None

    def test_agent_config_custom(self) -> None:
        config = AgentConfig(model="gpt-4", max_turns=5, system_prompt="Be helpful")
        assert config.model == "gpt-4"
        assert config.max_turns == 5
        assert config.system_prompt == "Be helpful"

    def test_turn_record_defaults(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="hello")
        turn = TurnRecord(turn_number=1, response_message=msg)
        assert turn.tool_calls == []
        assert turn.tool_results == []
        assert turn.usage is None
        assert turn.duration == 0.0

    def test_turn_stats_defaults(self) -> None:
        stats = TurnStats()
        assert stats.total_turns == 0
        assert stats.total_tool_calls == 0
        assert stats.total_duration == 0.0

    def test_agent_output_defaults(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="hi")
        output = AgentOutput(messages=[msg], response=msg, turns=[])
        assert output.stats == TurnStats()
        assert output.truncated is False
        assert output.error is None

    def test_message_entry_to_provider(self) -> None:
        entry = MessageEntry(role="user", content="hello", token_count=5)
        msg = message_entry_to_provider(entry)
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_message_entry_to_provider_with_tool_calls(self) -> None:
        entry = MessageEntry(
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        )
        msg = message_entry_to_provider(entry)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "c1"


class TestAgentLoop:
    @pytest.fixture
    def simple_response_provider(self) -> FakeProvider:
        return FakeProvider([make_text_response("Hello, I am an AI assistant.")])

    @pytest.fixture
    def tool_call_provider(self) -> FakeProvider:
        return FakeProvider([
            make_tool_call_response(),
            make_text_response("The result is ready."),
        ])

    @pytest.fixture
    def multi_tool_provider(self) -> FakeProvider:
        return FakeProvider([
            make_tool_call_response(tool_name="test_tool", args='{"a": 1}', tool_call_id="call_1"),
            make_tool_call_response(tool_name="test_tool", args='{"b": 2}', tool_call_id="call_2"),
            make_text_response("All tools executed."),
        ])

    async def test_simple_response(self, simple_response_provider: FakeProvider) -> None:
        loop = AgentLoop(provider=simple_response_provider)
        output = await loop.run("Hello")
        assert output.response.role == Role.ASSISTANT
        assert output.response.content == "Hello, I am an AI assistant."
        assert output.stats.total_turns == 1
        assert output.stats.total_tool_calls == 0

    async def test_single_tool_call(self, tool_call_provider: FakeProvider) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=tool_call_provider, tool_registry=registry)
        output = await loop.run("Use a tool")
        assert output.stats.total_turns == 2
        assert output.stats.total_tool_calls == 1
        assert output.response.content == "The result is ready."

    async def test_multi_tool_call(self, multi_tool_provider: FakeProvider) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=multi_tool_provider, tool_registry=registry)
        output = await loop.run("Use tools multiple times")
        assert output.stats.total_turns == 3
        assert output.stats.total_tool_calls == 2
        assert output.response.content == "All tools executed."

    async def test_placeholder_content_not_executed(self) -> None:
        provider = FakeProvider([
            make_tool_call_response(tool_name="write", args='{"path": "/tmp/x.py", "content": "(full code here)"}'),
            make_text_response("Fixed."),
        ])
        registry = ToolRegistry()
        registry.register(WriteTool())
        loop = AgentLoop(provider=provider, tool_registry=registry, config=AgentConfig(max_turns=3))
        output = await loop.run("write code")
        tool_messages = [m for m in output.messages if m.role == Role.TOOL]
        assert len(tool_messages) == 1
        assert "placeholder" in (tool_messages[0].content or "")
        assert "Missing required" not in (tool_messages[0].content or "")
        assert output.response.content == "Fixed."

    async def test_real_content_not_flagged_as_placeholder(self) -> None:
        content = (
            "<!DOCTYPE html>\n<html><body><h1>hello</h1></body></html>\n"
            "<script>var x = 1; function go() { return 2; }</script>"
        )
        provider = FakeProvider([
            make_tool_call_response(tool_name="write", args=json.dumps({"path": "/tmp/x.html", "content": content})),
            make_text_response("Done."),
        ])
        registry = ToolRegistry()
        registry.register(WriteTool())
        loop = AgentLoop(provider=provider, tool_registry=registry, config=AgentConfig(max_turns=3))
        output = await loop.run("write html")
        tool_messages = [m for m in output.messages if m.role == Role.TOOL]
        assert len(tool_messages) == 1
        assert "placeholder" not in (tool_messages[0].content or "")
        assert "bytes_written" in (tool_messages[0].content or "")

    async def test_max_turns_exceeded(self) -> None:
        provider = FakeProvider([
            make_tool_call_response(),
            make_tool_call_response(),
            make_tool_call_response(),
            make_tool_call_response(),
            make_tool_call_response(),
        ])
        registry = ToolRegistry()
        registry.register(FakeTool())
        config = AgentConfig(max_turns=3)
        loop = AgentLoop(provider=provider, tool_registry=registry, config=config)
        with pytest.raises(AgentTerminationError, match="Exceeded maximum turns"):
            await loop.run("Hello")

    async def test_repeated_tool_calls_inject_loop_hint(self) -> None:
        provider = FakeProvider([
            make_tool_call_response(),
            make_tool_call_response(),
            make_tool_call_response(),
            make_text_response("Final answer."),
        ])
        registry = ToolRegistry()
        registry.register(FakeTool())
        config = AgentConfig(max_repeated_tool_calls=1)
        loop = AgentLoop(provider=provider, tool_registry=registry, config=config)
        output = await loop.run("Loop check")
        contents = [m.content or "" for m in output.messages]
        assert any("repeating the same tool call" in c for c in contents)
        assert output.response.content == "Final answer."

    async def test_max_turns_returns_forced_final_answer(self) -> None:
        provider = FakeProvider([
            make_tool_call_response(),
            make_tool_call_response(),
            make_tool_call_response(),
            make_text_response("Final summary after limit."),
        ])
        registry = ToolRegistry()
        registry.register(FakeTool())
        config = AgentConfig(max_turns=3)
        loop = AgentLoop(provider=provider, tool_registry=registry, config=config)
        output = await loop.run("Work until limit")
        assert output.truncated is True
        assert output.response.content == "Final summary after limit."
        assert output.error is not None
        assert "Exceeded maximum turns" in output.error

    async def test_no_tool_registry(self, tool_call_provider: FakeProvider) -> None:
        loop = AgentLoop(provider=tool_call_provider, tool_registry=None)
        output = await loop.run("Use a tool")
        assert output.stats.total_turns == 2
        tool_messages = [m for m in output.messages if m.role == Role.TOOL]
        assert len(tool_messages) == 1
        assert "not available" in (tool_messages[0].content or "")

    async def test_system_prompt(self) -> None:
        provider = FakeProvider([make_text_response("Understood.")])
        config = AgentConfig(system_prompt="You are a helpful assistant.")
        loop = AgentLoop(provider=provider, config=config)
        output = await loop.run("Hello")
        assert output.response.content == "Understood."
        assert loop.config.system_prompt == "You are a helpful assistant."

    async def test_adaptive_prompt_small_model(self) -> None:
        captured: list[Message] = []

        class CaptureProvider(FakeProvider):
            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                captured.extend(request.messages)
                return make_text_response("Done.")

        loop = AgentLoop(provider=CaptureProvider([make_text_response("Done.")]), config=AgentConfig(model="gemma-3-270m-it"))
        await loop.run("hi")
        system = next(m for m in captured if m.role == Role.SYSTEM)
        assert "I am ByteCli" in (system.content or "")
        assert "Tool: read" in (system.content or "")

    async def test_adaptive_prompt_large_model(self) -> None:
        captured: list[Message] = []

        class CaptureProvider(FakeProvider):
            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                captured.extend(request.messages)
                return make_text_response("Done.")

        loop = AgentLoop(provider=CaptureProvider([make_text_response("Done.")]), config=AgentConfig(model="llama-3.1-8b-instruct"))
        await loop.run("hi")
        system = next(m for m in captured if m.role == Role.SYSTEM)
        assert "Rules:" in (system.content or "")
        assert "AI coding agent working directly" in (system.content or "")

    async def test_adaptive_prompt_uses_resolved_model(self) -> None:
        class ResolvingProvider(FakeProvider):
            async def resolve_model_name(self) -> str | None:
                return "phi-3-mini-4k"

        loop = AgentLoop(
            provider=ResolvingProvider([make_text_response("Done.")]),
            config=AgentConfig(model="local-model"),
        )
        await loop.run("hi")
        prompt = loop.effective_system_prompt()
        assert "I am ByteCli" in prompt
        assert "Only call a tool" in prompt

    async def test_resolved_model_returns_effective_model(self) -> None:
        class ResolvingProvider(FakeProvider):
            async def resolve_model_name(self) -> str | None:
                return "qwen2.5-coder-1.5b-instruct"

        loop = AgentLoop(
            provider=ResolvingProvider([make_text_response("Done.")]),
            config=AgentConfig(model="local-model"),
        )
        assert await loop.resolved_model() == "qwen2.5-coder-1.5b-instruct"

    async def test_resolved_model_falls_back_to_config_model(self) -> None:
        loop = AgentLoop(
            provider=FakeProvider([make_text_response("Done.")]),
            config=AgentConfig(model="some-model"),
        )
        assert await loop.resolved_model() == "some-model"

    async def test_custom_prompt_beats_auto_selection(self) -> None:
        captured: list[Message] = []

        class CaptureProvider(FakeProvider):
            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                captured.extend(request.messages)
                return make_text_response("Done.")

        config = AgentConfig(model="gemma-3-270m-it", system_prompt="My custom prompt")
        loop = AgentLoop(provider=CaptureProvider([make_text_response("Done.")]), config=config)
        await loop.run("hi")
        system = next(m for m in captured if m.role == Role.SYSTEM)
        assert system.content == "My custom prompt"

    async def test_provider_error(self) -> None:
        class ErrorProvider(FakeProvider):
            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                raise RuntimeError("Connection failed")

        loop = AgentLoop(provider=ErrorProvider([]))
        with pytest.raises(AgentLoopError, match="Provider error"):
            await loop.run("Hello")

    async def test_empty_response(self) -> None:
        provider = FakeProvider([
            CompletionResponse(id="r", model="m", choices=[], usage=None)
        ])
        loop = AgentLoop(provider=provider)
        with pytest.raises(AgentLoopError, match="Empty response"):
            await loop.run("Hello")

    async def test_broken_tool(self) -> None:
        provider = FakeProvider([
            make_tool_call_response(tool_name="broken_tool"),
            make_text_response("Tool result received."),
        ])
        registry = ToolRegistry()
        registry.register(BrokenTool())
        loop = AgentLoop(provider=provider, tool_registry=registry)
        output = await loop.run("Use broken tool")
        tool_messages = [m for m in output.messages if m.role == Role.TOOL]
        assert len(tool_messages) == 1
        assert "Error" in (tool_messages[0].content or "")

    async def test_unknown_tool_does_not_crash(self) -> None:
        provider = FakeProvider([
            make_tool_call_response(tool_name="mkdir_hack"),
            make_text_response("Recovered."),
        ])
        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=provider, tool_registry=registry)
        output = await loop.run("Make a folder")
        tool_messages = [m for m in output.messages if m.role == Role.TOOL]
        assert len(tool_messages) == 1
        assert "Unknown tool" in (tool_messages[0].content or "")
        assert "Available tools" in (tool_messages[0].content or "")
        assert "test_tool" in (tool_messages[0].content or "")
        assert output.response.content == "Recovered."

    async def test_unknown_tool_alias_resolves(self) -> None:
        provider = FakeProvider([
            make_tool_call_response(tool_name="fake_tool_alt"),
            make_text_response("Alias executed."),
        ])
        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=provider, tool_registry=registry)
        output = await loop.run("Do the thing")
        tool_messages = [m for m in output.messages if m.role == Role.TOOL]
        assert len(tool_messages) == 1
        assert "Unknown tool" not in (tool_messages[0].content or "")
        assert output.response.content == "Alias executed."

    async def test_available_tools_injected_into_system_prompt(self) -> None:
        captured: list[Message] = []

        class CaptureProvider(FakeProvider):
            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                captured.extend(request.messages)
                return make_text_response("Done.")

        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=CaptureProvider([make_text_response("Done.")]), tool_registry=registry)
        await loop.run("hi")
        system = next(m for m in captured if m.role == Role.SYSTEM)
        assert "Available tools:" in (system.content or "")
        assert "test_tool" in (system.content or "")

    async def test_memory_integration(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(session_id="test-session", workspace_dir=Path(tmpdir))
            provider = FakeProvider([make_text_response("Final answer.")])
            loop = AgentLoop(provider=provider, memory=memory)
            output = await loop.run("Hello")
            assert output.response.content == "Final answer."
            context = await memory.get_context()
            assert any(m.content == "Hello" for m in context)

    async def test_output_contains_turns(self, tool_call_provider: FakeProvider) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=tool_call_provider, tool_registry=registry)
        output = await loop.run("Use a tool")
        assert len(output.turns) == 2
        assert output.turns[0].turn_number == 1
        assert output.turns[1].turn_number == 2
        assert len(output.turns[0].tool_calls) == 1
        assert len(output.turns[1].tool_calls) == 0

    async def test_usage_tracking(self, tool_call_provider: FakeProvider) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        loop = AgentLoop(provider=tool_call_provider, tool_registry=registry)
        output = await loop.run("Use a tool")
        assert output.stats.total_prompt_tokens == 25
        assert output.stats.total_completion_tokens == 15

    async def test_event_bus_events(self) -> None:
        events: list[Event] = []
        bus = EventBus(auto_log=False)

        async def collector(event: Event) -> None:
            events.append(event)

        bus.subscribe("**", collector)

        provider = FakeProvider([make_text_response("Hi")])
        loop = AgentLoop(provider=provider, event_bus=bus)
        await loop.run("Hello")
        event_types = [e.type for e in events]
        assert "agent.turn.start" in event_types
        assert "agent.turn.end" in event_types

    async def test_run_stream(self) -> None:
        provider = FakeProvider([make_text_response("streamed response")])
        loop = AgentLoop(provider=provider)

        messages: list[Message] = []
        async for msg in loop.run_stream("Hello"):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].content == "streamed response"

    async def test_config_property(self) -> None:
        config = AgentConfig(model="gpt-4")
        loop = AgentLoop(provider=FakeProvider([make_text_response("ok")]), config=config)
        assert loop.config.model == "gpt-4"
        assert loop.config is config

    def test_parse_text_tool_call_inline(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        calls = loop._parse_text_tool_calls('Tool: read(path="test.txt")')
        assert len(calls) == 1
        assert calls[0].function["name"] == "read"
        args = json.loads(calls[0].function["arguments"])
        assert args == {"path": "test.txt"}

    def test_parse_text_tool_call_multiline(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = """I will read the file.
Tool: read
  path: /some/file.txt
  encoding: utf-8"""
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function["name"] == "read"
        args = json.loads(calls[0].function["arguments"])
        assert args == {"path": "/some/file.txt", "encoding": "utf-8"}

    def test_parse_text_tool_call_multiline_content(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = """Tool: write
  path: /tmp/app.py
  content:
    def main():
        print("hello")
"""
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function["name"] == "write"
        args = json.loads(calls[0].function["arguments"])
        assert args["path"] == "/tmp/app.py"
        assert "def main()" in args["content"]
        assert 'print("hello")' in args["content"]

    def test_parse_text_tool_call_unindented_content(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = """Tool: write
path: /tmp/app.py
content:
def main():
    print("hello")
"""
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert "def main()" in args["content"]

    def test_parse_text_tool_call_content_blank_lines_preserved(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = 'Tool: write\npath: /tmp/x.py\ncontent:\ndef a():\n\n    pass\n'
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert "def a():" in args["content"]
        assert "\n\n" in args["content"]

    def test_parse_text_tool_call_content_stops_at_prose(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = (
            "Tool: write\n  path: /tmp/app.py\n  content:\n    def main():\n"
            '        print("hi")\n\nI have written the file for you.'
        )
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert 'print("hi")' in args["content"]
        assert "I have written" not in args["content"]

    def test_parse_text_tool_call_flushed_when_prose_follows(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = "Tool: read\n  path: /a.txt\nDone reading."
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert args == {"path": "/a.txt"}

    def test_parse_text_tool_call_content_preserves_indentation(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = (
            "Tool: write\n  path: /tmp/app.py\n  content:\n"
            "    def greet():\n        print('hi')\n"
        )
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert args["content"] == "def greet():\n    print('hi')"

    def test_parse_text_tool_call_multiple(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = """First I'll read, then write.
Tool: read(path="a.txt")
Tool: write(path="b.txt", content="hello")"""
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].function["name"] == "read"
        assert calls[1].function["name"] == "write"

    def test_parse_text_tool_call_no_match(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        calls = loop._parse_text_tool_calls("I am just talking, no tools here.")
        assert len(calls) == 0

    def test_parse_text_tool_call_bare_name_ignored(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        calls = loop._parse_text_tool_calls("Tool: read")
        assert len(calls) == 0

    def test_parse_text_tool_call_empty_parens_ignored(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        calls = loop._parse_text_tool_calls("Tool: read()")
        assert len(calls) == 0

    def test_parse_text_tool_call_missing_args_in_block(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        calls = loop._parse_text_tool_calls("Tool: read\nTool: bash(command=\"echo hi\")")
        assert len(calls) == 1
        assert calls[0].function["name"] == "bash"

    def test_parse_css_properties_not_mistaken_for_arguments(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = (
            "Tool: write\n"
            "  path: E:/Downloads/snake.html\n"
            "  content:\n"
            "    <!DOCTYPE html>\n"
            "    <html>\n"
            "    <style>\n"
            "    body {\n"
            "      margin: 0;\n"
            "      padding: 0;\n"
            "    }\n"
            "    </style>\n"
            "</html>\n"
        )
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function["name"] == "write"
        args = json.loads(calls[0].function["arguments"])
        assert set(args.keys()) == {"path", "content"}
        content = args["content"]
        assert "margin: 0;" in content
        assert "padding: 0;" in content
        assert "body {" in content

    def test_parse_css_unindented_kept_in_content(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = (
            "Tool: write\n"
            "  path: /tmp/snake.html\n"
            "  content:\n"
            "<!DOCTYPE html>\n"
            "<style>\n"
            "body {\n"
            "  margin: 0;\n"
            "}\n"
            "</style>\n"
            "</html>\n"
        )
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert set(args.keys()) == {"path", "content"}
        assert "margin: 0;" in args["content"]

    def test_parse_json_content_not_mistaken_for_arguments(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = (
            "Tool: write\n"
            "  path: /tmp/data.json\n"
            "  content:\n"
            "    {\n"
            "      \"name\": \"test\",\n"
            "      \"color\": \"red\"\n"
            "    }\n"
        )
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert set(args.keys()) == {"path", "content"}
        assert '"name": "test"' in args["content"]

    def test_parse_new_arg_after_content_still_works(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        text = (
            "Tool: write\n"
            "  path: /tmp/x.py\n"
            "  content:\n"
            "    def foo():\n"
            "      pass\n"
            "  other_arg: hello\n"
        )
        calls = loop._parse_text_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function["arguments"])
        assert set(args.keys()) == {"path", "content", "other_arg"}
        assert args["other_arg"] == "hello"
        assert "def foo():" in args["content"]

    def test_strip_dangling_tool_calls(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("")]))
        assert loop._strip_dangling_tool_calls("Tool: read") == ""
        assert loop._strip_dangling_tool_calls("Sure!\nTool: read\nTool: bash()") == "Sure!"
        assert loop._strip_dangling_tool_calls("Tool: read\n  path: /a.txt") == "Tool: read\n  path: /a.txt"

    async def test_dangling_tool_call_removed_from_response(self) -> None:
        provider = FakeProvider([make_text_response("Tool: read")])
        loop = AgentLoop(provider=provider)
        output = await loop.run("hi")
        assert output.response.content == ""

    def test_tool_call_from_text_in_agent_loop(self) -> None:
        provider = FakeProvider([
            make_text_response('Tool: bash(command="echo hello")'),
            make_text_response("Done."),
        ])
        loop = AgentLoop(provider=provider, config=AgentConfig(max_turns=3))
        result = asyncio.run(loop.run("run a command"))
        assert len(result.turns) == 2
        assert len(result.turns[0].tool_calls) == 1
        assert result.turns[0].tool_calls[0].function["name"] == "bash"
        assert result.turns[0].tool_calls[0].function["arguments"] == '{"command": "echo hello"}'

    def test_multiline_write_text_tool_call_writes_file(self, tmp_path) -> None:
        from bytecli.tools.filesystem import WriteTool
        from bytecli.tools.registry import ToolRegistry

        target = tmp_path / "app.py"
        provider = FakeProvider([
            make_text_response(
                f"Tool: write\n  path: {target}\n  content:\n    def main():\n"
                '        print("hi")\n\nI wrote the file for you.'
            ),
            make_text_response("Done writing."),
        ])
        registry = ToolRegistry()
        registry.register(WriteTool())
        loop = AgentLoop(provider=provider, tool_registry=registry, config=AgentConfig(max_turns=3))
        result = asyncio.run(loop.run("create a python file"))
        assert len(result.turns) == 2
        assert target.read_text(encoding="utf-8") == 'def main():\n    print("hi")'
        assert "I wrote the file" not in target.read_text(encoding="utf-8")

    def test_extract_target_folder_windows(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        assert loop._extract_target_folder("save the game to this folder E:\\Downloads\\") == "E:\\Downloads"
        assert loop._extract_target_folder("put it in E:\\Projects\\web") == "E:\\Projects\\web"

    def test_extract_target_folder_unix(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        assert loop._extract_target_folder("save it to /home/user/games") == "/home/user/games"
        assert loop._extract_target_folder("store in ~/Downloads") == "~/Downloads"

    def test_extract_target_folder_none(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        assert loop._extract_target_folder("make a snake game") is None
        assert loop._extract_target_folder("hello there") is None

    def test_append_save_folder_hint_injects_folder(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        prompt = "You are ByteCli."
        result = loop._append_save_folder_hint(prompt, "save the game to E:\\Downloads\\")
        assert "E:\\Downloads" in result
        assert "ALWAYS use Tool: write" in result

    def test_append_save_folder_hint_no_folder(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        prompt = "You are ByteCli."
        assert loop._append_save_folder_hint(prompt, "just a chat") == prompt

    async def test_save_folder_hint_injected_into_system_prompt(self) -> None:
        captured: list[Message] = []

        class CaptureProvider(FakeProvider):
            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                captured.extend(request.messages)
                return make_text_response("Done.")

        loop = AgentLoop(provider=CaptureProvider([make_text_response("Done.")]))
        await loop.run("create a snake game and save it to E:\\Downloads\\")
        system = next(m for m in captured if m.role == Role.SYSTEM)
        assert "E:\\Downloads" in (system.content or "")
        assert "Tool: write" in (system.content or "")

    def test_accumulate_stream_concatenates_chunks(self) -> None:
        from collections.abc import AsyncIterator

        class StreamingProvider(Provider):
            def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
                async def gen() -> AsyncIterator[CompletionResponse]:
                    for text, finish in [("Hello ", None), ("world", None), ("", "stop")]:
                        yield CompletionResponse(
                            id="r1",
                            model="test",
                            choices=[Choice(
                                index=0,
                                delta=Message(role=Role.ASSISTANT, content=text),
                                finish_reason=finish,
                            )],
                        )

                return gen()

            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                raise AssertionError("chat should not be called when streaming works")

            async def list_models(self) -> list[ModelInfo]:
                return [ModelInfo(id="test")]

            def _build_headers(self) -> dict[str, str]:
                return {}

            def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
                return {}

            def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
                raise NotImplementedError

            def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
                raise NotImplementedError

        loop = AgentLoop(provider=StreamingProvider())
        result = asyncio.run(loop._accumulate_stream(
            CompletionRequest(model="test", messages=[Message(role=Role.USER, content="hi")]),
        ))
        assert result.choices[0].message.content == "Hello world"
        assert result.choices[0].finish_reason == "stop"

    def test_streaming_falls_back_to_chat_on_not_implemented(self) -> None:
        class ChatOnlyProvider(FakeProvider):
            def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
                raise NotImplementedError

        provider = ChatOnlyProvider([make_text_response("fallback answer")])
        loop = AgentLoop(provider=provider)
        result = asyncio.run(loop.run("hello"))
        assert result.response.content == "fallback answer"
        assert provider.call_count == 1

    def test_streaming_disabled_uses_chat_directly(self) -> None:
        class ChatOnlyProvider(FakeProvider):
            def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
                raise NotImplementedError

        provider = ChatOnlyProvider([make_text_response("direct answer")])
        loop = AgentLoop(provider=provider, config=AgentConfig(streaming=False))
        result = asyncio.run(loop.run("hello"))
        assert result.response.content == "direct answer"
        assert provider.call_count == 1

    def test_check_external_file_refs_no_refs(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        hint = loop._check_external_file_refs(
            "write",
            {"file_path": "game.html", "content": "<html><script>var x=1;</script></html>"},
            "saved",
        )
        assert hint is None

    def test_check_external_file_refs_warns(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        content = '<html><link href="style.css"><script src="app.js"></script></html>'
        hint = loop._check_external_file_refs(
            "write",
            {"file_path": "game.html", "content": content},
            "saved",
        )
        assert hint is not None
        assert "app.js" in hint
        assert "style.css" in hint
        assert "INLINE" in hint

    def test_check_external_file_refs_ignores_non_html_and_remote(self) -> None:
        loop = AgentLoop(provider=FakeProvider([make_text_response("Done.")]))
        assert loop._check_external_file_refs(
            "write",
            {"file_path": "app.py", "content": 'import "x.js"'},
            "saved",
        ) is None
        assert loop._check_external_file_refs(
            "write",
            {"file_path": "game.html", "content": '<script src="https://cdn.com/app.js"></script>'},
            "saved",
        ) is None

    async def test_external_ref_hint_appended_to_tool_result(self, tmp_path) -> None:
        from bytecli.tools.filesystem import WriteTool
        from bytecli.tools.registry import ToolRegistry

        target = tmp_path / "game.html"
        provider = FakeProvider([
            make_text_response(
                f"Tool: write\n  path: {target}\n  content:\n"
                '    <!DOCTYPE html>\n    <html>\n    <head>\n'
                '    <link rel="stylesheet" href="style.css">\n'
                '    </head>\n    <body>\n    <script src="app.js"></script>\n'
                '    </body>\n    </html>\n'
            ),
            make_text_response("Done."),
        ])
        registry = ToolRegistry()
        registry.register(WriteTool())
        loop = AgentLoop(provider=provider, tool_registry=registry, config=AgentConfig(max_turns=3))
        result = await loop.run("make a game")
        tool_msg = result.turns[0].tool_results[0]
        assert "app.js" in (tool_msg.content or "")
        assert "style.css" in (tool_msg.content or "")
        assert "WARNING" in (tool_msg.content or "")
