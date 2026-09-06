from pathlib import Path

import pytest
from platformdirs import user_config_dir

from bytecli.config.manager import ConfigManager
from bytecli.config.schema import ByteCliConfig, ProviderType
from bytecli.core.errors import ConfigValidationError


class TestConfigManager:
    def test_load_defaults(self):
        config_manager = ConfigManager()
        config = config_manager.load(env_override=False)
        assert isinstance(config, ByteCliConfig)
        assert config.provider.default == ProviderType.LM_STUDIO
        assert config.model.temperature == 0.7
        assert config.cli.theme.value == "auto"

    def test_load_defaults_auto_discover_none(self):
        config_manager = ConfigManager()
        config = config_manager.load(env_override=False)
        assert config.model.temperature == 0.7
        assert config.model.default == "local-model"

    def test_load_with_yaml(self, sample_yaml_config: Path):
        config_manager = ConfigManager()
        config = config_manager.load(config_path=sample_yaml_config, env_override=False)
        assert config.model.temperature == 0.1
        assert config.model.default == "test-model"
        assert config.provider.default == ProviderType.LM_STUDIO

    def test_load_with_toml(self, sample_toml_config: Path):
        config_manager = ConfigManager()
        config = config_manager.load(config_path=sample_toml_config, env_override=False)
        assert config.model.temperature == 0.2
        assert config.model.default == "toml-model"

    def test_load_with_json5(self, sample_json5_config: Path):
        config_manager = ConfigManager()
        config = config_manager.load(config_path=sample_json5_config, env_override=False)
        assert config.model.temperature == 0.3
        assert config.model.default == "json5-model"

    def test_cascade_yaml_overrides_defaults(self, sample_yaml_config: Path):
        config_manager = ConfigManager()
        config = config_manager.load(config_path=sample_yaml_config, env_override=False)
        assert config.model.temperature == 0.1
        assert config.provider.default == ProviderType.LM_STUDIO

    def test_get_raw_value(self, sample_yaml_config: Path):
        config_manager = ConfigManager()
        config_manager.load(config_path=sample_yaml_config, env_override=False)
        assert config_manager.get("model.temperature") == 0.1
        assert config_manager.get("model.default") == "test-model"
        assert config_manager.get("nonexistent.key", "fallback") == "fallback"

    def test_update_and_validate(self):
        config_manager = ConfigManager()
        config_manager.load(env_override=False)
        config_manager.update("model.temperature", 0.5)
        assert config_manager.config.model.temperature == 0.5
        assert config_manager.get("model.temperature") == 0.5

    def test_update_creates_nested(self):
        config_manager = ConfigManager()
        config_manager.load(env_override=False)
        config_manager.update("custom.section.value", "hello")
        assert config_manager.get("custom.section.value") == "hello"

    def test_invalid_update_raises(self):
        config_manager = ConfigManager()
        config_manager.load(env_override=False)
        with pytest.raises(ConfigValidationError):
            config_manager.update("model.temperature", 5.0)

    def test_reload(self):
        config_manager = ConfigManager()
        config_manager.load(env_override=False)
        config_manager.update("model.temperature", 0.5)
        assert config_manager.config.model.temperature == 0.5
        config_manager.reload()
        assert config_manager.config.model.temperature == 0.7

    def test_save_and_load_yaml(self, temp_dir: Path):
        config_manager = ConfigManager()
        config_manager.load(env_override=False)
        config_manager.update("model.default", "saved-model")
        save_path = temp_dir / "test_save.yaml"
        config_manager.save(save_path)
        assert save_path.exists()

        content = save_path.read_text(encoding="utf-8")
        assert "saved-model" in content
        assert "temperature" in content

        fresh_manager = ConfigManager()
        fresh_manager.load(config_path=save_path, env_override=False)
        assert fresh_manager.config.model.default == "saved-model"
        assert fresh_manager.config.model.temperature == 0.7

    def test_save_and_load_toml(self, temp_dir: Path):
        config_manager = ConfigManager()
        config_manager.load(env_override=False)
        save_path = temp_dir / "test_save.toml"
        config_manager.save(save_path)
        assert save_path.exists()

        fresh_manager = ConfigManager()
        fresh_manager.load(config_path=save_path, env_override=False)
        assert fresh_manager.config.model.temperature == 0.7

    def test_persist_dir(self):
        config_manager = ConfigManager()
        pdir = config_manager.persist_dir()
        assert isinstance(pdir, Path)
        assert "bytecli" in str(pdir)

    def test_config_dir(self):
        config_manager = ConfigManager()
        cdir = config_manager.config_dir()
        assert isinstance(cdir, Path)
        assert "bytecli" in str(cdir)
