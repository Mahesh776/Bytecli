import pytest

from bytecli.cache.backends import MemoryCache
from bytecli.cache.manager import CacheManager


class TestCacheManager:
    @pytest.mark.asyncio
    async def test_get_set(self, cache_manager: CacheManager):
        await cache_manager.set("test", "key1", "value1")
        result = await cache_manager.get("test", "key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing(self, cache_manager: CacheManager):
        result = await cache_manager.get("test", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_set_returns_cached(self, cache_manager: CacheManager):
        call_count = 0

        def factory() -> str:
            nonlocal call_count
            call_count += 1
            return "computed"

        result1 = await cache_manager.get_or_set("test", "key", factory, ttl=60)
        assert result1 == "computed"
        assert call_count == 1

        result2 = await cache_manager.get_or_set("test", "key", factory, ttl=60)
        assert result2 == "computed"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_get_or_set_expired(self, cache_manager: CacheManager):
        call_count = 0

        def factory() -> str:
            nonlocal call_count
            call_count += 1
            return "computed"

        await cache_manager.get_or_set("test", "key", factory, ttl=0)
        assert call_count == 1
        await cache_manager.get_or_set("test", "key", factory, ttl=0)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, cache_manager: CacheManager):
        await cache_manager.set("ns1", "key", "value1")
        await cache_manager.set("ns2", "key", "value2")
        assert await cache_manager.get("ns1", "key") == "value1"
        assert await cache_manager.get("ns2", "key") == "value2"

    @pytest.mark.asyncio
    async def test_delete(self, cache_manager: CacheManager):
        await cache_manager.set("test", "key", "value")
        await cache_manager.delete("test", "key")
        assert await cache_manager.get("test", "key") is None

    @pytest.mark.asyncio
    async def test_clear_namespace(self, cache_manager: CacheManager):
        await cache_manager.set("ns1", "k1", "v1")
        await cache_manager.set("ns2", "k1", "v2")
        await cache_manager.clear_namespace("ns1")
        assert await cache_manager.get("ns1", "k1") is None
        assert await cache_manager.get("ns2", "k1") == "v2"

    @pytest.mark.asyncio
    async def test_clear_all(self, cache_manager: CacheManager):
        await cache_manager.set("ns1", "k1", "v1")
        await cache_manager.set("ns2", "k1", "v2")
        await cache_manager.clear_all()
        assert await cache_manager.get("ns1", "k1") is None
        assert await cache_manager.get("ns2", "k1") is None

    @pytest.mark.asyncio
    async def test_size(self, cache_manager: CacheManager):
        await cache_manager.set("test", "k1", "v1")
        await cache_manager.set("test", "k2", "v2")
        assert await cache_manager.size("test") == 2

    @pytest.mark.asyncio
    async def test_total_size(self, cache_manager: CacheManager):
        await cache_manager.set("ns1", "k1", "v1")
        await cache_manager.set("ns2", "k1", "v2")
        total = await cache_manager.size()
        assert total >= 2

    @pytest.mark.asyncio
    async def test_register_new_namespace(self, cache_manager: CacheManager):
        cache_manager.register_namespace("custom", MemoryCache(), default_ttl=60)
        await cache_manager.set("custom", "key", "value")
        assert await cache_manager.get("custom", "key") == "value"

    @pytest.mark.asyncio
    async def test_default_ttl_used(self, cache_manager: CacheManager):
        cache_manager.register_namespace("tll_test", MemoryCache(), default_ttl=0)
        await cache_manager.set("tll_test", "key", "value")
        result = await cache_manager.get("tll_test", "key")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_override(self, cache_manager: CacheManager):
        cache_manager.register_namespace("default_ttl", MemoryCache(), default_ttl=0)
        await cache_manager.set("default_ttl", "key", "value", ttl=60)
        result = await cache_manager.get("default_ttl", "key")
        assert result == "value"
