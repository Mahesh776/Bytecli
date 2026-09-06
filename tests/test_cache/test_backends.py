import pytest

from bytecli.cache.backends import MemoryCache
from bytecli.core.errors import CacheError


class TestMemoryCache:
    @pytest.mark.asyncio
    async def test_get_set(self, memory_cache: MemoryCache):
        await memory_cache.set("key1", "value1")
        result = await memory_cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing(self, memory_cache: MemoryCache):
        result = await memory_cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_expired(self, memory_cache: MemoryCache):
        await memory_cache.set("key1", "value1", ttl=0)
        result = await memory_cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_not_expired(self, memory_cache: MemoryCache):
        await memory_cache.set("key1", "value1", ttl=60)
        result = await memory_cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_delete(self, memory_cache: MemoryCache):
        await memory_cache.set("key1", "value1")
        await memory_cache.delete("key1")
        result = await memory_cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_all(self, memory_cache: MemoryCache):
        await memory_cache.set("a:1", "v1")
        await memory_cache.set("b:1", "v2")
        await memory_cache.clear()
        assert await memory_cache.get("a:1") is None
        assert await memory_cache.get("b:1") is None

    @pytest.mark.asyncio
    async def test_clear_namespace(self, memory_cache: MemoryCache):
        await memory_cache.set("ns1:k1", "v1")
        await memory_cache.set("ns2:k1", "v2")
        await memory_cache.clear("ns1")
        assert await memory_cache.get("ns1:k1") is None
        assert await memory_cache.get("ns2:k1") == "v2"

    @pytest.mark.asyncio
    async def test_size(self, memory_cache: MemoryCache):
        assert await memory_cache.size() == 0
        await memory_cache.set("k1", "v1")
        assert await memory_cache.size() == 1
        await memory_cache.set("k2", "v2")
        assert await memory_cache.size() == 2

    @pytest.mark.asyncio
    async def test_size_expired_not_counted(self, memory_cache: MemoryCache):
        await memory_cache.set("k1", "v1", ttl=0)
        assert await memory_cache.size() == 0

    @pytest.mark.asyncio
    async def test_complex_values(self, memory_cache: MemoryCache):
        complex_value = {"nested": {"list": [1, 2, 3], "bool": True}, "number": 42}
        await memory_cache.set("complex", complex_value)
        result = await memory_cache.get("complex")
        assert result == complex_value

    @pytest.mark.asyncio
    async def test_overwrite(self, memory_cache: MemoryCache):
        await memory_cache.set("key", "old")
        await memory_cache.set("key", "new")
        assert await memory_cache.get("key") == "new"


class TestSQLiteCache:
    @pytest.mark.asyncio
    async def test_get_set(self, sqlite_cache):
        await sqlite_cache.set("key1", "value1")
        result = await sqlite_cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing(self, sqlite_cache):
        result = await sqlite_cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl(self, sqlite_cache):
        await sqlite_cache.set("key1", "value1", ttl=0)
        result = await sqlite_cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_cache):
        await sqlite_cache.set("key1", "value1")
        await sqlite_cache.delete("key1")
        result = await sqlite_cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_all(self, sqlite_cache):
        await sqlite_cache.set("a:1", "v1")
        await sqlite_cache.set("b:1", "v2")
        await sqlite_cache.clear()
        assert await sqlite_cache.get("a:1") is None
        assert await sqlite_cache.get("b:1") is None

    @pytest.mark.asyncio
    async def test_clear_namespace(self, sqlite_cache):
        await sqlite_cache.set("ns1:k1", "v1")
        await sqlite_cache.set("ns2:k1", "v2")
        await sqlite_cache.clear("ns1")
        assert await sqlite_cache.get("ns1:k1") is None
        assert await sqlite_cache.get("ns2:k1") == "v2"

    @pytest.mark.asyncio
    async def test_size(self, sqlite_cache):
        assert await sqlite_cache.size() == 0
        await sqlite_cache.set("k1", "v1")
        assert await sqlite_cache.size() == 1
        await sqlite_cache.set("k2", "v2")
        assert await sqlite_cache.size() == 2

    @pytest.mark.asyncio
    async def test_complex_values(self, sqlite_cache):
        complex_value = {"nested": {"list": [1, 2, 3], "bool": True}, "number": 42}
        await sqlite_cache.set("complex", complex_value)
        result = await sqlite_cache.get("complex")
        assert result == complex_value

    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir):
        db_path = temp_dir / "persist.db"
        from bytecli.cache.backends import SQLiteCache
        c1 = SQLiteCache(db_path, "test")
        await c1.set("persist_key", "persist_value")
        await c1.set("complex", {"nested": True})
        c1.close()

        c2 = SQLiteCache(db_path, "test")
        assert await c2.get("persist_key") == "persist_value"
        assert await c2.get("complex") == {"nested": True}
        c2.close()

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, sqlite_cache):
        import asyncio
        await sqlite_cache.set("concurrent", "shared")

        async def reader() -> str | None:
            return await sqlite_cache.get("concurrent")

        results = await asyncio.gather(*(reader() for _ in range(10)))
        assert all(r == "shared" for r in results)
