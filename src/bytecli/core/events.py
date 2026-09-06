import fnmatch
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class Event:
    type: str
    timestamp: float
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("Event type must be a non-empty string")
        if not isinstance(self.data, dict):
            raise ValueError("Event data must be a dict")


Listener = Callable[[Event], Awaitable[None]]


class EventBus:
    APP_START = "app.start"
    APP_STOP = "app.stop"
    APP_ERROR = "app.error"

    SESSION_START = "session.start"
    SESSION_END = "session.end"
    SESSION_RESUME = "session.resume"

    AGENT_TURN_START = "agent.turn.start"
    AGENT_TURN_END = "agent.turn.end"
    AGENT_THINKING = "agent.thinking"
    AGENT_ACTION = "agent.action"
    AGENT_ERROR = "agent.error"

    TOOL_EXECUTION_START = "tool.execution.start"
    TOOL_EXECUTION_END = "tool.execution.end"
    TOOL_EXECUTION_ERROR = "tool.execution.error"

    PROVIDER_REQUEST_START = "provider.request.start"
    PROVIDER_REQUEST_END = "provider.request.end"
    PROVIDER_STREAM_CHUNK = "provider.stream.chunk"

    COMMAND_EXECUTION_START = "command.execution.start"
    COMMAND_EXECUTION_END = "command.execution.end"

    PLUGIN_LOAD = "plugin.load"
    PLUGIN_UNLOAD = "plugin.unload"
    PLUGIN_ERROR = "plugin.error"

    MEMORY_COMPACTION_START = "memory.compaction.start"
    MEMORY_COMPACTION_END = "memory.compaction.end"

    CONFIG_RELOAD = "config.reload"
    CONFIG_CHANGE = "config.change"

    def __init__(
        self,
        auto_log: bool = True,
        event_filter: set[str] | None = None,
    ) -> None:
        self._listeners: dict[str, list[tuple[int, Listener]]] = defaultdict(list)
        self._auto_log = auto_log
        self._event_filter: set[str] | None = event_filter

    def subscribe(self, topic: str, listener: Listener, priority: int = 100) -> None:
        expanded = self._expand_topic(topic)
        for t in expanded:
            self._listeners[t].append((priority, listener))
            self._listeners[t].sort(key=lambda x: x[0])

    def unsubscribe(self, topic: str, listener: Listener) -> None:
        expanded = self._expand_topic(topic)
        for t in expanded:
            self._listeners[t] = [(p, item) for p, item in self._listeners[t] if item is not listener]
            if not self._listeners[t]:
                del self._listeners[t]

    async def publish(self, event: Event) -> None:
        if self._auto_log and self._should_log(event.type):
            safe_data = self._sanitize_for_log(event.data)
            logger.debug("Event: {} | data={}", event.type, safe_data)

        matching: list[tuple[int, Listener]] = []
        for topic, listeners in self._listeners.items():
            if topic == event.type or fnmatch.fnmatch(event.type, topic):
                matching.extend(listeners)

        matching.sort(key=lambda x: x[0])

        for priority, listener in matching:
            try:
                await listener(event)
            except Exception:
                logger.exception("Listener failed for event '{}' at priority {}", event.type, priority)

    def listener_count(self) -> int:
        return sum(len(v) for v in self._listeners.values())

    def _expand_topic(self, topic: str) -> list[str]:
        if "**" in topic:
            return ["**"]
        if "*" in topic or "?" in topic:
            return [topic]
        return [topic]

    def _should_log(self, event_type: str) -> bool:
        if self._event_filter is not None:
            return event_type in self._event_filter
        return True

    def _sanitize_for_log(self, data: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(data)
        for key in list(sanitized.keys()):
            if any(sensitive in key.lower() for sensitive in ("key", "token", "secret", "password", "auth")):
                sanitized[key] = "***"
            elif isinstance(sanitized[key], str) and len(sanitized[key]) > 500:
                sanitized[key] = sanitized[key][:500] + "..."
        return sanitized

    @staticmethod
    def create_event(type: str, data: dict[str, Any] | None = None) -> Event:
        return Event(type=type, timestamp=time.time(), data=data or {})
