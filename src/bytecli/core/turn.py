from dataclasses import dataclass, field
from typing import Any

from bytecli.providers.types import Message, Role, ToolCall, Usage


@dataclass
class AgentConfig:
    model: str = "local-model"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_turns: int = 25
    max_repeated_tool_calls: int = 3
    system_prompt: str | None = None
    streaming: bool = True


@dataclass
class TurnRecord:
    turn_number: int
    response_message: Message
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[Message] = field(default_factory=list)
    usage: Usage | None = None
    duration: float = 0.0


@dataclass
class TurnStats:
    total_turns: int = 0
    total_tool_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_duration: float = 0.0


@dataclass
class AgentOutput:
    messages: list[Message]
    response: Message
    turns: list[TurnRecord]
    stats: TurnStats = field(default_factory=TurnStats)
    truncated: bool = False
    error: str | None = None


def message_entry_to_provider(msg: Any) -> Message:
    tool_calls: list[ToolCall] | None = None
    if msg.tool_calls:
        tool_calls = [ToolCall(**tc) if isinstance(tc, dict) else tc for tc in msg.tool_calls]
    return Message(
        role=Role(msg.role),
        content=msg.content,
        tool_calls=tool_calls,
        tool_call_id=msg.tool_call_id,
        name=msg.name,
    )
