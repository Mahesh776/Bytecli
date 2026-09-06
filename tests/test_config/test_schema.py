import pytest
from pydantic import ValidationError

from bytecli.config.schema import (
    ByteCliConfig,
    CLIConfig,
    CacheConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    ProviderConfig,
    ProviderEndpointConfig,
    ProviderType,
    SecurityConfig,
    ThemeType,
    WorkspaceConfig,
    is_small_model,
    parse_model_params,
    select_system_prompt,
)


class TestProviderEndpointConfig:
    def test_defaults(self):
        config = ProviderEndpointConfig()
        assert config.base_url == "http://localhost:1234/v1"
        assert config.timeout == 60.0
        assert config.max_retries == 3

    def test_custom(self):
        config = ProviderEndpointConfig(base_url="http://other:8080/v1", timeout=30.0, api_key="test-key")
        assert config.base_url == "http://other:8080/v1"
        assert config.timeout == 30.0
        assert config.api_key == "test-key"


class TestProviderConfig:
    def test_defaults(self):
        config = ProviderConfig()
        assert config.default == ProviderType.LM_STUDIO
        assert config.lm_studio.base_url == "http://localhost:1234/v1"
        assert config.ollama.base_url == "http://localhost:11434/v1"
        assert config.lm_studio.timeout == 300.0
        assert config.ollama.timeout == 300.0
        assert config.vllm.timeout == 300.0
        assert config.openai.timeout == 60.0

    def test_change_default(self):
        config = ProviderConfig(default=ProviderType.OLLAMA)
        assert config.default == ProviderType.OLLAMA


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig()
        assert config.default == "local-model"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.top_p == 0.95
        assert config.system_prompt is None

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            ModelConfig(temperature=2.1)
        ModelConfig(temperature=0.0)
        ModelConfig(temperature=2.0)

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            ModelConfig(max_tokens=0)
        with pytest.raises(ValidationError):
            ModelConfig(max_tokens=999999)
        ModelConfig(max_tokens=1)
        ModelConfig(max_tokens=131072)


class TestCLIConfig:
    def test_defaults(self):
        config = CLIConfig()
        assert config.theme == ThemeType.AUTO
        assert config.multi_line is True
        assert config.history_size == 1000

    def test_theme_enum(self):
        assert ThemeType.LIGHT.value == "light"
        assert ThemeType.DARK.value == "dark"
        assert ThemeType.AUTO.value == "auto"


class TestMemoryConfig:
    def test_defaults(self):
        config = MemoryConfig()
        assert config.max_context_tokens == 32000
        assert config.compaction_threshold == 0.8
        assert config.session_persistence is True


class TestSecurityConfig:
    def test_defaults(self):
        config = SecurityConfig()
        assert config.ask_before_tool is False
        assert "rm -rf /" in config.blocked_commands

    def test_allowed_commands(self):
        config = SecurityConfig(allowed_commands=["git", "npm"])
        assert "git" in config.allowed_commands


class TestLoggingConfig:
    def test_defaults(self):
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.json_format is False

    def test_invalid_level(self):
        config = LoggingConfig(level="DEBUG")
        assert config.level == "DEBUG"


class TestCacheConfig:
    def test_defaults(self):
        config = CacheConfig()
        assert config.enabled is True
        assert config.default_ttl_provider == 300


class TestWorkspaceConfig:
    def test_defaults(self):
        config = WorkspaceConfig()
        assert config.auto_detect is True
        assert config.index_depth == 3
        assert ".git" in config.ignored_dirs


class TestByteCliConfig:
    def test_defaults(self):
        config = ByteCliConfig()
        assert config.provider.default == ProviderType.LM_STUDIO
        assert config.model.temperature == 0.7
        assert config.cli.theme == ThemeType.AUTO
        assert config.memory.max_context_tokens == 32000
        assert config.cache.enabled is True
        assert config.workspace.auto_detect is True

    def test_nested_override(self):
        config = ByteCliConfig(
            model=ModelConfig(temperature=0.1, default="my-model"),
            provider=ProviderConfig(default=ProviderType.OLLAMA),
        )
        assert config.model.temperature == 0.1
        assert config.model.default == "my-model"
        assert config.provider.default == ProviderType.OLLAMA
        assert config.cli.theme == ThemeType.AUTO

    def test_from_dict(self):
        data = {
            "model": {"temperature": 0.5, "default": "custom"},
            "provider": {"default": "anthropic"},
        }
        config = ByteCliConfig.model_validate(data)
        assert config.model.temperature == 0.5
        assert config.model.default == "custom"
        assert config.provider.default == ProviderType.ANTHROPIC

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValidationError):
            ByteCliConfig.model_validate({"model": {"temperature": 5.0}})

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            ByteCliConfig(logging=LoggingConfig(level="INVALID"))


class TestPromptSelection:
    def test_parse_model_params(self) -> None:
        assert parse_model_params("gemma-3-270m-it") == 0.27
        assert parse_model_params("llama-3.1-8b-instruct") == 8.0
        assert parse_model_params("local-model") is None

    def test_is_small_model_param_based(self) -> None:
        assert is_small_model("gemma-3-270m-it") is True
        assert is_small_model("qwen2.5-coder-1.5b") is True
        assert is_small_model("llama-3.2-1b-instruct") is True
        assert is_small_model("llama-3.1-8b-instruct") is False
        assert is_small_model("qwen2.5-coder-32b") is False

    def test_is_small_model_keyword_based(self) -> None:
        assert is_small_model("phi-3-mini-4k") is True
        assert is_small_model("tinyllama") is True
        assert is_small_model("mistral-7b") is False

    def test_select_system_prompt_small(self) -> None:
        prompt = select_system_prompt("gemma-3-270m-it")
        assert "Only call a tool" in prompt
        assert "I am ByteCli" in prompt

    def test_select_system_prompt_large(self) -> None:
        prompt = select_system_prompt("llama-3.1-8b-instruct")
        assert "Rules:" in prompt
        assert "Example:" not in prompt

    def test_select_system_prompt_custom_wins(self) -> None:
        prompt = select_system_prompt("gemma-3-270m-it", custom_prompt="custom")
        assert prompt == "custom"

    def test_system_prompt_instructs_following_skills(self) -> None:
        for prompt in (
            select_system_prompt("gemma-3-270m-it"),
            select_system_prompt("llama-3.1-8b-instruct"),
        ):
            assert "skill instructions" in prompt
