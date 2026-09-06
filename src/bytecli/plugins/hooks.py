from collections.abc import Awaitable, Callable
from typing import Any

HOOK_APP_STARTUP = "app.startup"
HOOK_APP_SHUTDOWN = "app.shutdown"
HOOK_APP_BEFORE_INPUT = "app.before_input"
HOOK_APP_AFTER_OUTPUT = "app.after_output"
HOOK_APP_ON_ERROR = "app.on_error"

HOOK_AGENT_BEFORE_TURN = "agent.before_turn"
HOOK_AGENT_AFTER_TURN = "agent.after_turn"
HOOK_AGENT_BEFORE_THINK = "agent.before_think"
HOOK_AGENT_AFTER_THINK = "agent.after_think"
HOOK_AGENT_BEFORE_ACT = "agent.before_act"
HOOK_AGENT_AFTER_ACT = "agent.after_act"

HOOK_TOOL_BEFORE_EXECUTE = "tool.before_execute"
HOOK_TOOL_AFTER_EXECUTE = "tool.after_execute"
HOOK_TOOL_ON_ERROR = "tool.on_error"

HOOK_PROVIDER_BEFORE_REQUEST = "provider.before_request"
HOOK_PROVIDER_AFTER_REQUEST = "provider.after_request"
HOOK_PROVIDER_ON_STREAM_CHUNK = "provider.on_stream_chunk"

HOOK_MEMORY_BEFORE_STORE = "memory.before_store"
HOOK_MEMORY_AFTER_STORE = "memory.after_store"
HOOK_MEMORY_BEFORE_RETRIEVE = "memory.before_retrieve"
HOOK_MEMORY_AFTER_RETRIEVE = "memory.after_retrieve"
HOOK_MEMORY_ON_COMPACT = "memory.on_compact"

HOOK_SESSION_START = "session.start"
HOOK_SESSION_END = "session.end"
HOOK_SESSION_BEFORE_COMMAND = "session.before_command"
HOOK_SESSION_AFTER_COMMAND = "session.after_command"

HOOK_CONFIG_BEFORE_RELOAD = "config.before_reload"
HOOK_CONFIG_AFTER_RELOAD = "config.after_reload"
HOOK_CONFIG_ON_CHANGE = "config.on_change"

ALL_HOOKS = [
    HOOK_APP_STARTUP,
    HOOK_APP_SHUTDOWN,
    HOOK_APP_BEFORE_INPUT,
    HOOK_APP_AFTER_OUTPUT,
    HOOK_APP_ON_ERROR,
    HOOK_AGENT_BEFORE_TURN,
    HOOK_AGENT_AFTER_TURN,
    HOOK_AGENT_BEFORE_THINK,
    HOOK_AGENT_AFTER_THINK,
    HOOK_AGENT_BEFORE_ACT,
    HOOK_AGENT_AFTER_ACT,
    HOOK_TOOL_BEFORE_EXECUTE,
    HOOK_TOOL_AFTER_EXECUTE,
    HOOK_TOOL_ON_ERROR,
    HOOK_PROVIDER_BEFORE_REQUEST,
    HOOK_PROVIDER_AFTER_REQUEST,
    HOOK_PROVIDER_ON_STREAM_CHUNK,
    HOOK_MEMORY_BEFORE_STORE,
    HOOK_MEMORY_AFTER_STORE,
    HOOK_MEMORY_BEFORE_RETRIEVE,
    HOOK_MEMORY_AFTER_RETRIEVE,
    HOOK_MEMORY_ON_COMPACT,
    HOOK_SESSION_START,
    HOOK_SESSION_END,
    HOOK_SESSION_BEFORE_COMMAND,
    HOOK_SESSION_AFTER_COMMAND,
    HOOK_CONFIG_BEFORE_RELOAD,
    HOOK_CONFIG_AFTER_RELOAD,
    HOOK_CONFIG_ON_CHANGE,
]

HookHandler = Callable[..., Awaitable[None]]


class HookRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, list[tuple[int, str, HookHandler]]] = {}

    def register(self, hook: str, handler: HookHandler, plugin_name: str = "", priority: int = 100) -> None:
        if hook not in self._handlers:
            self._handlers[hook] = []
        self._handlers[hook].append((priority, plugin_name, handler))
        self._handlers[hook].sort(key=lambda x: x[0])

    def unregister_all(self, plugin_name: str) -> None:
        for hook in list(self._handlers.keys()):
            self._handlers[hook] = [(p, n, h) for p, n, h in self._handlers[hook] if n != plugin_name]
            if not self._handlers[hook]:
                del self._handlers[hook]

    async def execute(self, hook: str, **kwargs: Any) -> None:
        handlers = self._handlers.get(hook)
        if not handlers:
            return
        for priority, plugin_name, handler in handlers:
            await handler(**kwargs)

    def list_hooks(self) -> list[str]:
        return list(self._handlers.keys())

    def handlers_for_hook(self, hook: str) -> list[tuple[int, str]]:
        return [(p, n) for p, n, _ in self._handlers.get(hook, [])]

    def clear(self) -> None:
        self._handlers.clear()
