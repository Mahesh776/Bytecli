import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str) -> Any | None: ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self, namespace: str | None = None) -> None: ...

    @abstractmethod
    async def size(self) -> int: ...


class MemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    async def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value, expires = self._store[key]
        if expires is not None and time.time() > expires:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires: float | None = None
        if ttl is not None and ttl <= 0:
            expires = time.time()
        elif ttl is not None:
            expires = time.time() + ttl
        self._store[key] = (value, expires)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._store.clear()
        else:
            prefix = f"{namespace}:"
            self._store = {k: v for k, v in self._store.items() if not k.startswith(prefix)}

    async def size(self) -> int:
        now = time.time()
        self._store = {k: v for k, v in self._store.items() if v[1] is None or now <= v[1]}
        return len(self._store)


class SQLiteCache(CacheBackend):
    def __init__(self, db_path: Path, table_name: str = "cache") -> None:
        self._db_path = db_path
        self._table_name = table_name
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS [{self._table_name}] (key TEXT PRIMARY KEY, value BLOB, expires REAL)"
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._table_name}_expires ON [{self._table_name}](expires)")
        conn.commit()

    async def get(self, key: str) -> Any | None:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                f"SELECT value, expires FROM [{self._table_name}] WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            expires = row["expires"]
            if expires is not None and time.time() > expires:
                conn.execute(f"DELETE FROM [{self._table_name}] WHERE key = ?", (key,))
                conn.commit()
                return None
            try:
                return json.loads(row["value"])
            except (TypeError, json.JSONDecodeError):
                return row["value"]

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            conn = self._get_conn()
            expires: float | None = None
            if ttl is not None and ttl <= 0:
                expires = time.time()
            elif ttl is not None:
                expires = time.time() + ttl
            conn.execute(
                f"INSERT OR REPLACE INTO [{self._table_name}] (key, value, expires) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires),
            )
            conn.commit()

    async def delete(self, key: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(f"DELETE FROM [{self._table_name}] WHERE key = ?", (key,))
            conn.commit()

    async def clear(self, namespace: str | None = None) -> None:
        with self._lock:
            conn = self._get_conn()
            if namespace is None:
                conn.execute(f"DELETE FROM [{self._table_name}]")
            else:
                conn.execute(
                    f"DELETE FROM [{self._table_name}] WHERE key LIKE ?",
                    (f"{namespace}:%",),
                )
            conn.commit()

    async def size(self) -> int:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(f"SELECT COUNT(*) as cnt FROM [{self._table_name}]")
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
