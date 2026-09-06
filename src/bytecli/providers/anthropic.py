from collections.abc import AsyncIterator
from typing import Any

from bytecli.core.errors import ProviderError
from bytecli.providers.base_http import BaseHTTPProvider
from bytecli.providers.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    ModelInfo,
    Role,
    Usage,
)


class AnthropicProvider(BaseHTTPProvider):
    def __init__(
        self,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(base_url, api_key, timeout, max_retries)
        self._api_version = "2023-06-01"

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": self._api_version,
        }
        return headers

    def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
        system_msg = None
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_msg = msg.content
            else:
                entry: dict[str, Any] = {"role": msg.role.value, "content": msg.content or ""}
                messages.append(entry)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        if system_msg is not None:
            payload["system"] = system_msg
        if request.temperature != 0.7:
            payload["temperature"] = request.temperature
        if request.top_p != 0.95:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        payload.update(request.extra)
        return payload

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        message = Message(role=Role.ASSISTANT, content=content or None)
        choice = Choice(index=0, message=message, finish_reason=data.get("stop_reason"))

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            )

        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=[choice],
            usage=usage,
            created=data.get("created"),
            raw=data,
        )

    def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
        if not line:
            return None
        import json

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        event_type = data.get("type", "")
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                content = delta.get("text", "")
                message = Message(role=Role.ASSISTANT, content=content)
                choice = Choice(index=0, delta=message)
                return CompletionResponse(
                    id=data.get("id", ""),
                    model="",
                    choices=[choice],
                )
        elif event_type == "message_start":
            msg = data.get("message", {})
            return CompletionResponse(
                id=msg.get("id", ""),
                model=msg.get("model", ""),
                choices=[],
            )
        elif event_type == "message_stop":
            return None  # Signal done

        return None

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_chat_payload(request)
        payload["stream"] = False
        data = await self._request("POST", "/messages", payload)
        return self._parse_response(data)

    async def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        payload = self._build_chat_payload(request)
        payload["stream"] = True
        async for line in self._stream_request("/messages", payload):
            if line.startswith("{"):
                chunk = self._parse_stream_chunk(line)
                if chunk is not None:
                    yield chunk

    async def list_models(self) -> list[ModelInfo]:
        data = await self._request("GET", "/models")
        models: list[ModelInfo] = []
        for m in data.get("data", []):
            models.append(
                ModelInfo(
                    id=m.get("id", ""),
                    owned_by=m.get("owned_by", "anthropic"),
                    created=m.get("created", 0),
                )
            )
        return models

    async def validate_connection(self) -> bool:
        try:
            await self._request("GET", "/models")
            return True
        except ProviderError:
            return False
