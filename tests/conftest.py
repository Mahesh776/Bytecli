import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest

from bytecli.cache.backends import MemoryCache, SQLiteCache
from bytecli.cache.manager import CacheManager
from bytecli.config.loader import ConfigLoader
from bytecli.config.manager import ConfigManager
from bytecli.config.schema import ByteCliConfig
from bytecli.core.events import EventBus


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus(auto_log=False)


@pytest.fixture
def memory_cache() -> MemoryCache:
    return MemoryCache()


@pytest.fixture
def sqlite_cache(temp_dir: Path) -> Generator[SQLiteCache, None, None]:
    cache = SQLiteCache(temp_dir / "test.db")
    yield cache
    cache.close()


@pytest.fixture
def cache_manager(temp_dir: Path) -> Generator[CacheManager, None, None]:
    cm = CacheManager(persist_dir=temp_dir)
    yield cm
    cm.close()


@pytest.fixture
def config_manager() -> ConfigManager:
    return ConfigManager()


@pytest.fixture
def sample_yaml_config(temp_dir: Path) -> Path:
    path = temp_dir / ".bytecli.yaml"
    path.write_text("model:\n  temperature: 0.1\n  default: test-model\n")
    return path


@pytest.fixture
def sample_toml_config(temp_dir: Path) -> Path:
    path = temp_dir / ".bytecli.toml"
    path.write_text('[model]\ntemperature = 0.2\ndefault = "toml-model"\n')
    return path


@pytest.fixture
def sample_json5_config(temp_dir: Path) -> Path:
    path = temp_dir / ".bytecli.json5"
    path.write_text('{\n  model: {\n    temperature: 0.3,\n    default: "json5-model",\n  },\n}\n')
    return path
