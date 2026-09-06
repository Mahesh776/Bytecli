from typing import Any


class ByteCliError(Exception):
    def __init__(self, message: str = "", cause: BaseException | None = None) -> None:
        super().__init__(message)
        self._message = message
        if cause is not None:
            self.__cause__ = cause

    @property
    def message(self) -> str:
        return self._message


class ConfigError(ByteCliError):
    pass


class ConfigNotFoundError(ConfigError):
    def __init__(self, path: str, cause: BaseException | None = None) -> None:
        super().__init__(f"Configuration file not found: {path}", cause=cause)
        self.path = path


class ConfigValidationError(ConfigError):
    def __init__(self, errors: list[dict[str, Any]], cause: BaseException | None = None) -> None:
        messages = [f"{e.get('loc', 'unknown')}: {e.get('msg', e.get('message', str(e)))}" for e in errors]
        super().__init__("Configuration validation failed:\n" + "\n".join(messages), cause=cause)
        self.errors = errors


class ConfigMergeError(ConfigError):
    def __init__(self, source: str, target: str, detail: str, cause: BaseException | None = None) -> None:
        super().__init__(f"Cannot merge config from '{source}' into '{target}': {detail}", cause=cause)
        self.source = source
        self.target = target


class ProviderError(ByteCliError):
    pass


class ProviderConnectionError(ProviderError):
    def __init__(self, provider: str, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Failed to connect to provider '{provider}'{': ' + detail if detail else ''}", cause=cause)
        self.provider = provider


class ProviderAuthError(ProviderError):
    def __init__(self, provider: str, detail: str = "", cause: BaseException | None = None) -> None:
        msg = f"Authentication failed for provider '{provider}'{': ' + detail if detail else ''}"
        super().__init__(msg, cause=cause)
        self.provider = provider


class ProviderRateLimitError(ProviderError):
    def __init__(self, provider: str, retry_after: float | None = None, cause: BaseException | None = None) -> None:
        msg = f"Rate limited by provider '{provider}'"
        if retry_after:
            msg += f", retry after {retry_after:.0f}s"
        super().__init__(msg, cause=cause)
        self.provider = provider
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    def __init__(self, provider: str, timeout: float, cause: BaseException | None = None) -> None:
        super().__init__(f"Request to provider '{provider}' timed out after {timeout}s", cause=cause)
        self.provider = provider
        self.timeout = timeout


class ProviderModelError(ProviderError):
    def __init__(self, provider: str, model: str, detail: str = "", cause: BaseException | None = None) -> None:
        msg = f"Model '{model}' not available on provider '{provider}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, cause=cause)
        self.provider = provider
        self.model = model


class ToolError(ByteCliError):
    pass


class ToolNotFoundError(ToolError):
    def __init__(self, tool_name: str, cause: BaseException | None = None) -> None:
        super().__init__(f"Tool not found: {tool_name}", cause=cause)
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    def __init__(self, tool_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Tool '{tool_name}' execution failed{': ' + detail if detail else ''}", cause=cause)
        self.tool_name = tool_name


class ToolTimeoutError(ToolError):
    def __init__(self, tool_name: str, timeout: float, cause: BaseException | None = None) -> None:
        super().__init__(f"Tool '{tool_name}' timed out after {timeout}s", cause=cause)
        self.tool_name = tool_name
        self.timeout = timeout


class ToolPermissionError(ToolError):
    def __init__(self, tool_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Permission denied for tool '{tool_name}'{': ' + detail if detail else ''}", cause=cause)
        self.tool_name = tool_name


class ToolValidationError(ToolError):
    def __init__(self, tool_name: str, errors: list[str], cause: BaseException | None = None) -> None:
        super().__init__(f"Invalid arguments for tool '{tool_name}': {'; '.join(errors)}", cause=cause)
        self.tool_name = tool_name
        self.errors = errors


class CommandError(ByteCliError):
    pass


class CommandNotFoundError(CommandError):
    def __init__(self, command_name: str, cause: BaseException | None = None) -> None:
        super().__init__(f"Command not found: {command_name}", cause=cause)
        self.command_name = command_name


class CommandExecutionError(CommandError):
    def __init__(self, command_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Command '{command_name}' execution failed{': ' + detail if detail else ''}", cause=cause)
        self.command_name = command_name


class CommandPermissionError(CommandError):
    def __init__(self, command_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        msg = f"Permission denied for command '{command_name}'{': ' + detail if detail else ''}"
        super().__init__(msg, cause=cause)
        self.command_name = command_name


class PluginError(ByteCliError):
    pass


class PluginLoadError(PluginError):
    def __init__(self, plugin_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Failed to load plugin '{plugin_name}'{': ' + detail if detail else ''}", cause=cause)
        self.plugin_name = plugin_name


class PluginHookError(PluginError):
    def __init__(self, plugin_name: str, hook_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        msg = f"Plugin '{plugin_name}' hook '{hook_name}' failed{': ' + detail if detail else ''}"
        super().__init__(msg, cause=cause)
        self.plugin_name = plugin_name
        self.hook_name = hook_name


class PluginDependencyError(PluginError):
    def __init__(self, plugin_name: str, dependency: str, cause: BaseException | None = None) -> None:
        super().__init__(f"Plugin '{plugin_name}' missing dependency: {dependency}", cause=cause)
        self.plugin_name = plugin_name
        self.dependency = dependency


class SkillError(ByteCliError):
    pass


class SkillNotFoundError(SkillError):
    def __init__(self, skill_name: str, cause: BaseException | None = None) -> None:
        super().__init__(f"Skill not found: {skill_name}", cause=cause)
        self.skill_name = skill_name


class SkillLoadError(SkillError):
    def __init__(self, skill_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Failed to load skill '{skill_name}'{': ' + detail if detail else ''}", cause=cause)
        self.skill_name = skill_name


class SkillValidationError(SkillError):
    def __init__(self, skill_name: str, errors: list[str], cause: BaseException | None = None) -> None:
        super().__init__(f"Skill '{skill_name}' validation failed: {'; '.join(errors)}", cause=cause)
        self.skill_name = skill_name
        self.errors = errors


class MemoryError(ByteCliError):
    pass


class MemoryStoreError(MemoryError):
    def __init__(self, store_name: str, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Memory store '{store_name}' error{': ' + detail if detail else ''}", cause=cause)
        self.store_name = store_name


class MemoryCompactionError(MemoryError):
    def __init__(self, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Memory compaction failed{': ' + detail if detail else ''}", cause=cause)


class CacheError(ByteCliError):
    def __init__(self, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Cache error{': ' + detail if detail else ''}", cause=cause)


class AgentError(ByteCliError):
    pass


class AgentLoopError(AgentError):
    def __init__(self, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Agent loop error{': ' + detail if detail else ''}", cause=cause)


class AgentContextOverflowError(AgentError):
    def __init__(self, max_tokens: int, cause: BaseException | None = None) -> None:
        super().__init__(f"Agent context overflow: maximum {max_tokens} tokens exceeded", cause=cause)
        self.max_tokens = max_tokens


class AgentTerminationError(AgentError):
    def __init__(self, reason: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Agent terminated{': ' + reason if reason else ''}", cause=cause)
        self.reason = reason


class WorkspaceError(ByteCliError):
    def __init__(self, detail: str = "", cause: BaseException | None = None) -> None:
        super().__init__(f"Workspace error{': ' + detail if detail else ''}", cause=cause)
