from abc import ABC, abstractmethod

from bytecli.memory.types import MemoryEntry, MemoryLevel, MemoryStats


class MemoryStore(ABC):
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None: ...

    @abstractmethod
    async def retrieve(self, key: str, level: MemoryLevel | None = None) -> MemoryEntry | None: ...

    @abstractmethod
    async def delete(self, key: str, level: MemoryLevel | None = None) -> None: ...

    @abstractmethod
    async def clear(self, level: MemoryLevel | None = None) -> None: ...

    @abstractmethod
    async def stats(self, level: MemoryLevel | None = None) -> MemoryStats: ...

    @abstractmethod
    async def search(self, query: str, level: MemoryLevel | None = None, limit: int = 10) -> list[MemoryEntry]: ...
