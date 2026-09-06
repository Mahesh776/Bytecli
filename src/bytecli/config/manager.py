import copy
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir, user_data_dir

from bytecli.config.defaults import DEFAULTS
from bytecli.config.loader import ConfigLoader
from bytecli.config.schema import ByteCliConfig
from bytecli.core.errors import ConfigError, ConfigValidationError

_BYTECLI_DIR_NAME = "bytecli"


class ConfigManager:
    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root
        self._config: ByteCliConfig | None = None
        self._raw: dict[str, Any] = {}

    @property
    def config(self) -> ByteCliConfig:
        if self._config is None:
            self.load()
        return self._config  # type: ignore[return-value]

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    def load(
        self,
        config_path: Path | None = None,
        env_override: bool = True,
    ) -> ByteCliConfig:
        configs: list[dict[str, Any]] = [copy.deepcopy(DEFAULTS)]

        if config_path:
            user_config = ConfigLoader.load_file(config_path)
            configs.append(user_config)
        else:
            global_config = self._find_global_config()
            if global_config:
                configs.append(ConfigLoader.load_file(global_config))

            local_config = self._find_local_config()
            if local_config:
                configs.append(ConfigLoader.load_file(local_config))

            project_config = self._find_project_config()
            if project_config:
                configs.append(ConfigLoader.load_file(project_config))

        merged = ConfigLoader.merge(*configs)

        if env_override:
            env_config = ConfigLoader.load_env_vars("BYTECLI_")
            if env_config:
                merged = ConfigLoader.merge(merged, env_config)

        self._raw = merged
        self._config = self._validate(merged)
        return self._config

    def reload(self) -> ByteCliConfig:
        return self.load()

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        current: Any = self._raw
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default
        return current

    def update(self, key_path: str, value: Any) -> None:
        keys = key_path.split(".")
        current = self._raw
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        self._config = self._validate(self._raw)

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = self._find_local_config() or self._default_local_path()
        ConfigLoader.dump_file(self._raw, path)

    def persist_dir(self) -> Path:
        return Path(user_data_dir(_BYTECLI_DIR_NAME, ensure_exists=True))

    def config_dir(self) -> Path:
        return Path(user_config_dir(_BYTECLI_DIR_NAME, ensure_exists=True))

    def _find_global_config(self) -> Path | None:
        config_dir = Path(user_config_dir(_BYTECLI_DIR_NAME, ensure_exists=True))
        for name in ("config.yaml", "config.yml", "config.toml", "config.json5", "config.json"):
            candidate = config_dir / name
            if candidate.exists():
                return candidate
        return None

    def _find_local_config(self) -> Path | None:
        config_names = (
            ".bytecli.yaml",
            ".bytecli.yml",
            ".bytecli.toml",
            ".bytecli.json5",
            ".bytecli.json",
            "bytecli.yaml",
        )
        for name in config_names:
            candidate = Path.cwd() / name
            if candidate.exists():
                return candidate
        return None

    def _find_project_config(self) -> Path | None:
        if self._project_root is None:
            return None
        for name in (".bytecli.yaml", ".bytecli.yml", ".bytecli.toml", ".bytecli.json5", ".bytecli.json"):
            candidate = self._project_root / name
            if candidate.exists():
                return candidate
        return None

    def _default_local_path(self) -> Path:
        return Path.cwd() / ".bytecli.yaml"

    def _validate(self, data: dict[str, Any]) -> ByteCliConfig:
        try:
            return ByteCliConfig.model_validate(data)
        except Exception as e:
            from pydantic import ValidationError

            if isinstance(e, ValidationError):
                errors = [
                    {
                        "loc": ".".join(str(x) for x in err.get("loc", [])),
                        "msg": err.get("msg", err.get("message", str(err))),
                    }
                    for err in e.errors()
                ]
                raise ConfigValidationError(errors) from e
            raise ConfigError(f"Config validation failed: {e}") from e
