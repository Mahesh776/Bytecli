import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml

try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore[assignment]

import json5

from bytecli.core.errors import ConfigError, ConfigNotFoundError


class ConfigFormat(ABC):
    extensions: ClassVar[set[str]]

    @abstractmethod
    def load(self, content: str) -> dict[str, Any]: ...

    @abstractmethod
    def dump(self, data: dict[str, Any]) -> str: ...


class YamlConfig(ConfigFormat):
    extensions: ClassVar[set[str]] = {".yaml", ".yml"}

    def load(self, content: str) -> dict[str, Any]:
        try:
            result = yaml.safe_load(content)
            if result is None:
                return {}
            if not isinstance(result, dict):
                raise ConfigError(f"YAML root must be a mapping, got {type(result).__name__}")
            return result
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}") from e

    def dump(self, data: dict[str, Any]) -> str:
        result: str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return result


class TomlConfig(ConfigFormat):
    extensions: ClassVar[set[str]] = {".toml"}

    def load(self, content: str) -> dict[str, Any]:
        if tomllib is None:
            raise ConfigError("TOML support requires Python 3.11+")
        try:
            result = tomllib.loads(content)
            if not isinstance(result, dict):
                raise ConfigError(f"TOML root must be a table, got {type(result).__name__}")
            return result
        except Exception as e:
            raise ConfigError(f"Invalid TOML: {e}") from e

    def dump(self, data: dict[str, Any]) -> str:
        try:
            import tomli_w

            result: str = tomli_w.dumps(data)
            return result
        except ImportError:
            lines: list[str] = ["# ByteCli Configuration\n"]
            self._toml_serialize(lines, data, "")
            return "\n".join(lines)

    def _toml_serialize(self, lines: list[str], data: dict[str, Any], prefix: str) -> None:
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                lines.append(f"\n[{full_key}]")
                self._toml_serialize(lines, value, full_key)
            elif isinstance(value, list):
                lines.append(f"{key} = {json.dumps(value)}")
            elif isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, str):
                if any(c in value for c in '"\\\n\t'):
                    lines.append(f'{key} = """{value}"""')
                else:
                    lines.append(f'{key} = "{value}"')
            elif value is None:
                lines.append(f"{key} = ''")
            else:
                lines.append(f"{key} = {value}")


class Json5Config(ConfigFormat):
    extensions: ClassVar[set[str]] = {".json5", ".json"}

    def load(self, content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            return {}
        try:
            result = json5.loads(content)
            if result is None:
                return {}
            if not isinstance(result, dict):
                raise ConfigError(f"JSON root must be an object, got {type(result).__name__}")
            return result
        except ValueError as e:
            raise ConfigError(f"Invalid JSON5: {e}") from e

    def dump(self, data: dict[str, Any]) -> str:
        return json.dumps(data, indent=2, default=str)


class ConfigLoader:
    _formats: ClassVar[dict[str, ConfigFormat]] = {}

    @classmethod
    def register(cls, fmt: ConfigFormat) -> None:
        for ext in fmt.extensions:
            cls._formats[ext] = fmt

    @classmethod
    def load_file(cls, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ConfigNotFoundError(str(path))
        ext = path.suffix.lower()
        fmt = cls._formats.get(ext)
        if fmt is None:
            raise ConfigError(f"Unsupported config format: {ext}. Supported: {list(cls._formats.keys())}")
        content = path.read_text(encoding="utf-8")
        return fmt.load(content)

    @classmethod
    def dump_file(cls, data: dict[str, Any], path: Path) -> None:
        ext = path.suffix.lower()
        fmt = cls._formats.get(ext)
        if fmt is None:
            raise ConfigError(f"Unsupported config format: {ext}. Supported: {list(cls._formats.keys())}")
        content = fmt.dump(data)
        path.write_text(content, encoding="utf-8")

    @classmethod
    def load_env_vars(cls, prefix: str = "BYTECLI_") -> dict[str, Any]:
        result: dict[str, Any] = {}
        env_pattern = re.compile(rf"^{re.escape(prefix)}(.+)$")

        for env_key, env_value in os.environ.items():
            match = env_pattern.match(env_key)
            if not match:
                continue
            if not env_value:
                continue
            config_key = match.group(1).lower().replace("__", ".")
            keys = config_key.split(".")
            current = result
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    current[k] = cls._parse_env_value(env_value)
                else:
                    if k not in current:
                        current[k] = {}
                    if not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]

        return result

    @classmethod
    def merge(cls, *configs: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for config in configs:
            cls._deep_merge(result, config)
        return result

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        if value.lower() in ("none", "null", ""):
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        if value.startswith("{") and value.endswith("}"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return value

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigLoader._deep_merge(base[key], value)
            else:
                base[key] = value


ConfigLoader.register(YamlConfig())
ConfigLoader.register(TomlConfig())
ConfigLoader.register(Json5Config())
