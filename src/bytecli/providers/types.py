from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    id: str
    type: str = "function"
    function: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class Choice:
    index: int = 0
    message: Message | None = None
    delta: Message | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    stop: list[str] | None = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResponse:
    id: str
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    created: int | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class Delta:
    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True)
class StreamChunk:
    id: str
    model: str
    choices: list[Choice]
    created: int | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    object: str = "model"
    owned_by: str = ""
    created: int = 0
