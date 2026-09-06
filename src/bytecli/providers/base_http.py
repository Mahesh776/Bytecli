from collections.abc import AsyncIterator
from typing import Any

import httpx

from bytecli.core.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from bytecli.logging.logger import get_logger
from bytecli.providers.base import Provider

logger = get_logger("bytecli.providers")


class BaseHTTPProvider(Provider):
    def __init__(
        self,
        base_url: str = "",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(base_url, api_key, timeout, max_retries)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        headers = self._build_headers()
        url = f"{self.base_url}{path}"

        for attempt in range(self.max_retries):
            try:
                response = await client.request(
                    method,
                    url,
                    json=json_data,
                    headers=headers,
                )
            except httpx.TimeoutException as e:
                if attempt < self.max_retries - 1:
                    continue
                raise ProviderTimeoutError(self.__class__.__name__, self.timeout) from e
            except httpx.ConnectError as e:
                raise ProviderConnectionError(self.__class__.__name__, str(e)) from e
            except httpx.HTTPError as e:
                raise ProviderConnectionError(self.__class__.__name__, str(e)) from e

            if response.status_code == 429:
                if attempt < self.max_retries - 1:
                    continue
                raise ProviderRateLimitError(self.__class__.__name__)
            if response.status_code == 401:
                raise ProviderAuthError(self.__class__.__name__, "Invalid API key")
            if response.status_code >= 500:
                if attempt < self.max_retries - 1:
                    continue
                raise ProviderError(
                    f"{self.__class__.__name__}: Server error: {response.status_code}",
                )

            if response.status_code >= 400:
                body = response.text[:500]
                raise ProviderError(
                    f"{self.__class__.__name__}: HTTP {response.status_code}: {body}",
                )

            result: dict[str, Any] = response.json()
            return result

        raise ProviderError(f"{self.__class__.__name__}: Max retries exceeded")

    async def _stream_request(
        self,
        path: str,
        json_data: dict[str, Any],
    ) -> AsyncIterator[str]:
        client = self._get_client()
        headers = self._build_headers()
        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"
        url = f"{self.base_url}{path}"

        try:
            async with client.stream(
                "POST",
                url,
                json=json_data,
                headers=headers,
            ) as response:
                if response.status_code == 401:
                    raise ProviderAuthError(self.__class__.__name__, "Invalid API key")
                if response.status_code == 429:
                    raise ProviderRateLimitError(self.__class__.__name__)
                if response.status_code >= 400:
                    raw = await response.aread()
                    body = raw.decode("utf-8", errors="replace")[:500]
                    raise ProviderError(
                        f"{self.__class__.__name__}: HTTP {response.status_code}: {body}",
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        yield line[6:]
                    elif line.startswith("data:"):
                        yield line[5:]

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(self.__class__.__name__, self.timeout) from e
        except httpx.ConnectError as e:
            raise ProviderConnectionError(self.__class__.__name__, str(e)) from e
        except httpx.HTTPError as e:
            raise ProviderConnectionError(self.__class__.__name__, str(e)) from e

    async def validate_connection(self) -> bool:
        try:
            await self._request("GET", "/models")
            return True
        except ProviderError:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
