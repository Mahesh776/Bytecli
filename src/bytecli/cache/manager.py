from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from bytecli.cache.backends import CacheBackend, MemoryCache, SQLiteCache

T = TypeVar("T")


class CacheManager:
    _NAMESPACE_SEPARATOR = ":"

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._backends: dict[str, tuple[CacheBackend, int | None]] = {}
        if persist_dir is not None:
            self.register_namespace("provider", SQLiteCache(persist_dir / "provider.db"), default_ttl=300)
            self.register_namespace("workspace", SQLiteCache(persist_dir / "workspace.db"), default_ttl=60)
        else:
            self.register_namespace("provider", MemoryCache(), default_ttl=300)
            self.register_namespace("workspace", MemoryCache(), default_ttl=60)
        self.register_namespace("tool", MemoryCache(), default_ttl=30)
        self.register_namespace("session", MemoryCache(), default_ttl=None)
        self.register_namespace("_internal", MemoryCache(), default_ttl=None)

    def register_namespace(self, namespace: str, backend: CacheBackend, default_ttl: int | None = None) -> None:
        self._backends[namespace] = (backend, default_ttl)

    async def get(self, namespace: str, key: str) -> Any | None:
        backend, _ = self._get_backend(namespace)
        full_key = f"{namespace}{self._NAMESPACE_SEPARATOR}{key}"
        return await backend.get(full_key)

    async def set(self, namespace: str, key: str, value: Any, ttl: int | None = None) -> None:
        backend, default_ttl = self._get_backend(namespace)
        full_key = f"{namespace}{self._NAMESPACE_SEPARATOR}{key}"
        effective_ttl = ttl if ttl is not None else default_ttl
        await backend.set(full_key, value, ttl=effective_ttl)

    async def delete(self, namespace: str, key: str) -> None:
        backend, _ = self._get_backend(namespace)
        full_key = f"{namespace}{self._NAMESPACE_SEPARATOR}{key}"
        await backend.delete(full_key)

    async def clear_namespace(self, namespace: str) -> None:
        backend, _ = self._get_backend(namespace)
        await backend.clear(namespace)

    async def clear_all(self) -> None:
        for backend, _ in self._backends.values():
            await backend.clear()

    async def get_or_set(self, namespace: str, key: str, factory: Callable[[], T], ttl: int | None = None) -> T:
        cached = await self.get(namespace, key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]
        value = factory()
        await self.set(namespace, key, value, ttl=ttl)
        return value

    async def size(self, namespace: str | None = None) -> int:
        if namespace is not None:
            backend, _ = self._get_backend(namespace)
            return await backend.size()
        total = 0
        for backend, _ in self._backends.values():
            total += await backend.size()
        return total

    def _get_backend(self, namespace: str) -> tuple[CacheBackend, int | None]:
        if namespace not in self._backends:
            self.register_namespace(namespace, MemoryCache())
        return self._backends[namespace]

    def close(self) -> None:
        for backend, _ in self._backends.values():
            if hasattr(backend, "close"):
                backend.close()
