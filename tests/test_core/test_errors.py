from bytecli.core.errors import (
    AgentContextOverflowError,
    AgentError,
    AgentLoopError,
    AgentTerminationError,
    ByteCliError,
    CacheError,
    CommandError,
    CommandExecutionError,
    CommandNotFoundError,
    CommandPermissionError,
    ConfigError,
    ConfigMergeError,
    ConfigNotFoundError,
    ConfigValidationError,
    MemoryCompactionError,
    MemoryError,
    MemoryStoreError,
    PluginDependencyError,
    PluginError,
    PluginHookError,
    PluginLoadError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderModelError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SkillError,
    SkillLoadError,
    SkillNotFoundError,
    SkillValidationError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
    WorkspaceError,
)


def test_all_errors_inherit_from_base():
    all_errors = [
        ByteCliError(),
        ConfigError(),
        ConfigNotFoundError("/path/to/config"),
        ConfigValidationError([{"loc": "model.temperature", "msg": "too high"}]),
        ConfigMergeError("a.yaml", "b.yaml", "key conflict"),
        ProviderError(),
        ProviderConnectionError("lm_studio", "connection refused"),
        ProviderAuthError("openai", "invalid key"),
        ProviderRateLimitError("openai", retry_after=30.0),
        ProviderTimeoutError("lm_studio", 60.0),
        ProviderModelError("ollama", "llama3", "not found"),
        ToolError(),
        ToolNotFoundError("read_file"),
        ToolExecutionError("bash", "exit code 1"),
        ToolTimeoutError("bash", 30.0),
        ToolPermissionError("bash", "not allowed"),
        ToolValidationError("write", ["path required"]),
        CommandError(),
        CommandNotFoundError("/help"),
        CommandExecutionError("/test", "failed"),
        CommandPermissionError("/admin"),
        PluginError(),
        PluginLoadError("my-plugin", "import error"),
        PluginHookError("my-plugin", "on_message", "timeout"),
        PluginDependencyError("my-plugin", "requests"),
        SkillError(),
        SkillNotFoundError("python-pro"),
        SkillLoadError("python-pro", "missing system.md"),
        SkillValidationError("python-pro", ["missing skill.json"]),
        MemoryError(),
        MemoryStoreError("session", "write failed"),
        MemoryCompactionError("too many tokens"),
        CacheError("connection failed"),
        AgentError(),
        AgentLoopError("infinite loop detected"),
        AgentContextOverflowError(32000),
        AgentTerminationError("user cancelled"),
        WorkspaceError("no project detected"),
    ]

    for err in all_errors:
        assert isinstance(err, ByteCliError), f"{type(err).__name__} is not a ByteCliError"
        assert isinstance(err, Exception), f"{type(err).__name__} is not an Exception"


def test_error_hierarchy():
    assert issubclass(ConfigNotFoundError, ConfigError)
    assert issubclass(ConfigError, ByteCliError)
    assert issubclass(ByteCliError, Exception)

    assert issubclass(ProviderConnectionError, ProviderError)
    assert issubclass(ProviderAuthError, ProviderError)
    assert issubclass(ProviderRateLimitError, ProviderError)
    assert issubclass(ProviderTimeoutError, ProviderError)
    assert issubclass(ProviderModelError, ProviderError)

    assert issubclass(ToolNotFoundError, ToolError)
    assert issubclass(ToolExecutionError, ToolError)
    assert issubclass(ToolTimeoutError, ToolError)
    assert issubclass(ToolPermissionError, ToolError)
    assert issubclass(ToolValidationError, ToolError)

    assert issubclass(CommandNotFoundError, CommandError)
    assert issubclass(CommandExecutionError, CommandError)
    assert issubclass(CommandPermissionError, CommandError)

    assert issubclass(PluginLoadError, PluginError)
    assert issubclass(PluginHookError, PluginError)
    assert issubclass(PluginDependencyError, PluginError)


def test_error_message():
    err = ConfigNotFoundError("/path/to/config.yaml")
    assert "not found" in err.message
    assert "/path/to/config.yaml" in err.message


def test_error_chaining():
    cause = ValueError("original error")
    err = ProviderConnectionError("lm_studio", "wrapped", cause=cause)
    assert err.__cause__ is cause


def test_config_validation_error():
    errors = [{"loc": ("model", "temperature"), "msg": "too high"}]
    err = ConfigValidationError(errors)
    assert "temperature" in err.message
    assert "too high" in err.message
    assert err.errors == errors


def test_provider_rate_limit_error():
    err = ProviderRateLimitError("openai", retry_after=30.0)
    assert err.retry_after == 30.0
    assert "30" in err.message


def test_agent_context_overflow():
    err = AgentContextOverflowError(32000)
    assert err.max_tokens == 32000
    assert "32000" in err.message


def test_provider_model_error():
    err = ProviderModelError("ollama", "llama3", "model not installed")
    assert err.provider == "ollama"
    assert err.model == "llama3"
    assert "llama3" in err.message
    assert "not installed" in err.message


def test_errors_picklable():
    import pickle

    err = ProviderConnectionError("lm_studio", "refused")
    data = pickle.dumps(err)
    loaded = pickle.loads(data)
    assert loaded.message == err.message
    assert loaded.provider == err.provider
