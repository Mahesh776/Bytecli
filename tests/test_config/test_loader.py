import os
from pathlib import Path

import pytest

from bytecli.config.loader import (
    ConfigFormat,
    ConfigLoader,
    Json5Config,
    TomlConfig,
    YamlConfig,
)
from bytecli.core.errors import ConfigError, ConfigNotFoundError


class TestYamlConfig:
    def test_load(self):
        loader = YamlConfig()
        data = loader.load("model:\n  temperature: 0.5\n  name: test\n")
        assert data == {"model": {"temperature": 0.5, "name": "test"}}

    def test_load_empty(self):
        loader = YamlConfig()
        assert loader.load("") == {}

    def test_load_invalid(self):
        loader = YamlConfig()
        with pytest.raises(ConfigError, match="YAML"):
            loader.load("{{invalid")

    def test_dump_and_reload(self):
        loader = YamlConfig()
        data = {"provider": {"default": "ollama"}}
        dumped = loader.dump(data)
        reloaded = loader.load(dumped)
        assert reloaded == data

    def test_extensions(self):
        assert ".yaml" in YamlConfig().extensions
        assert ".yml" in YamlConfig().extensions


class TestTomlConfig:
    def test_load(self):
        loader = TomlConfig()
        data = loader.load('[model]\ntemperature = 0.5\nname = "test"\n')
        assert data == {"model": {"temperature": 0.5, "name": "test"}}

    def test_load_empty(self):
        loader = TomlConfig()
        result = loader.load("")
        assert isinstance(result, dict)

    def test_load_invalid(self):
        loader = TomlConfig()
        with pytest.raises(ConfigError, match="TOML"):
            loader.load("{{invalid")

    def test_dump(self):
        loader = TomlConfig()
        data = {"provider": {"default": "ollama"}}
        dumped = loader.dump(data)
        assert "ollama" in dumped
        assert "provider" in dumped

    def test_extensions(self):
        assert ".toml" in TomlConfig().extensions


class TestJson5Config:
    def test_load(self):
        loader = Json5Config()
        data = loader.load('{\n  model: {\n    temperature: 0.5,\n    name: "test",\n  },\n}')
        assert data == {"model": {"temperature": 0.5, "name": "test"}}

    def test_load_empty(self):
        loader = Json5Config()
        assert loader.load("") == {}
        assert loader.load("null") == {}

    def test_load_invalid(self):
        loader = Json5Config()
        with pytest.raises(ConfigError, match="JSON"):
            loader.load("{invalid")

    def test_load_with_comments(self):
        loader = Json5Config()
        data = loader.load('{\n  // comment\n  model: "test",\n}')
        assert data == {"model": "test"}

    def test_dump(self):
        loader = Json5Config()
        data = {"provider": {"default": "ollama"}}
        dumped = loader.dump(data)
        assert "ollama" in dumped

    def test_extensions(self):
        assert ".json5" in Json5Config().extensions
        assert ".json" in Json5Config().extensions


class TestConfigLoader:
    def test_register_format(self):
        class TestFormat(ConfigFormat):
            extensions = {".test"}
            def load(self, content: str) -> dict:
                return {"from": "test"}
            def dump(self, data: dict) -> str:
                return "test"

        ConfigLoader.register(TestFormat())
        assert ".test" in ConfigLoader._formats

    def test_load_yaml_file(self, sample_yaml_config: Path):
        data = ConfigLoader.load_file(sample_yaml_config)
        assert data["model"]["temperature"] == 0.1
        assert data["model"]["default"] == "test-model"

    def test_load_toml_file(self, sample_toml_config: Path):
        data = ConfigLoader.load_file(sample_toml_config)
        assert data["model"]["temperature"] == 0.2
        assert data["model"]["default"] == "toml-model"

    def test_load_json5_file(self, sample_json5_config: Path):
        data = ConfigLoader.load_file(sample_json5_config)
        assert data["model"]["temperature"] == 0.3
        assert data["model"]["default"] == "json5-model"

    def test_load_nonexistent_file(self):
        with pytest.raises(ConfigNotFoundError):
            ConfigLoader.load_file(Path("/nonexistent/config.yaml"))

    def test_load_unsupported_format(self, temp_dir: Path):
        path = temp_dir / "config.unsupported"
        path.write_text("{}")
        with pytest.raises(ConfigError, match="Unsupported"):
            ConfigLoader.load_file(path)

    def test_env_vars(self):
        os.environ["BYTECLI_MODEL__TEMPERATURE"] = "0.9"
        os.environ["BYTECLI_MODEL__DEFAULT"] = "env-model"
        os.environ["BYTECLI_PROVIDER__DEFAULT"] = "ollama"
        try:
            env_config = ConfigLoader.load_env_vars("BYTECLI_")
            assert env_config["model"]["temperature"] == 0.9
            assert env_config["model"]["default"] == "env-model"
            assert env_config["provider"]["default"] == "ollama"
        finally:
            del os.environ["BYTECLI_MODEL__TEMPERATURE"]
            del os.environ["BYTECLI_MODEL__DEFAULT"]
            del os.environ["BYTECLI_PROVIDER__DEFAULT"]

    def test_env_var_bool_parsing(self):
        os.environ["BYTECLI_CLI__MULTI_LINE"] = "false"
        os.environ["BYTECLI_MEMORY__SESSION_PERSISTENCE"] = "true"
        try:
            env_config = ConfigLoader.load_env_vars("BYTECLI_")
            assert env_config["cli"]["multi_line"] is False
            assert env_config["memory"]["session_persistence"] is True
        finally:
            del os.environ["BYTECLI_CLI__MULTI_LINE"]
            del os.environ["BYTECLI_MEMORY__SESSION_PERSISTENCE"]

    def test_env_var_list_parsing(self):
        os.environ["BYTECLI_MODEL__STOP_SEQUENCES"] = '["\\n", "```"]'
        try:
            env_config = ConfigLoader.load_env_vars("BYTECLI_")
            assert env_config["model"]["stop_sequences"] == ["\n", "```"]
        finally:
            del os.environ["BYTECLI_MODEL__STOP_SEQUENCES"]

    def test_merge_simple(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        merged = ConfigLoader.merge(base, override)
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self):
        base = {"model": {"temperature": 0.7, "default": "base"}}
        override = {"model": {"temperature": 0.5}}
        merged = ConfigLoader.merge(base, override)
        assert merged["model"]["temperature"] == 0.5
        assert merged["model"]["default"] == "base"

    def test_merge_deep_nested(self):
        base = {"provider": {"lm_studio": {"base_url": "http://a:1234/v1", "timeout": 60}}}
        override = {"provider": {"lm_studio": {"timeout": 120}}}
        merged = ConfigLoader.merge(base, override)
        assert merged["provider"]["lm_studio"]["base_url"] == "http://a:1234/v1"
        assert merged["provider"]["lm_studio"]["timeout"] == 120

    def test_merge_cascade(self):
        defaults = {"model": {"temperature": 0.7, "default": "default-model"}}
        user = {"model": {"temperature": 0.5}}
        project = {"model": {"default": "project-model"}}
        env = {"model": {"temperature": 0.1}}
        merged = ConfigLoader.merge(defaults, user, project, env)
        assert merged["model"]["temperature"] == 0.1
        assert merged["model"]["default"] == "project-model"
