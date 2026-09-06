from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from bytecli.providers.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ModelInfo,
)


class Provider(ABC):
    def __init__(
        self,
        base_url: str = "",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    async def chat(self, request: CompletionRequest) -> CompletionResponse: ...

    @abstractmethod
    def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]: ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    async def resolve_model_name(self) -> str | None:
        return None

    async def validate_connection(self) -> bool:
        try:
            await self.list_models()
            return True
        except Exception:
            return False

    @abstractmethod
    def _build_headers(self) -> dict[str, str]: ...

    @abstractmethod
    def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]: ...

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse: ...

    @abstractmethod
    def _parse_stream_chunk(self, line: str) -> CompletionResponse | None: ...

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role.value}
            if msg.content is not None:
                d["content"] = msg.content
            if msg.tool_calls:
                d["tool_calls"] = [
                    {"id": tc.id, "type": tc.type, "function": tc.function}
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name
            result.append(d)
        return result
