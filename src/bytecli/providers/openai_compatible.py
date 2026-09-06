from collections.abc import AsyncIterator
from typing import Any

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


class OpenAICompatibleProvider(BaseHTTPProvider):
    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._messages_to_dicts(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": request.stream,
        }
        if request.stop:
            payload["stop"] = request.stop
        payload.update(request.extra)
        return payload

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        choices = []
        for c in data.get("choices", []):
            msg_data = c.get("message", {})
            message = Message(
                role=Role(msg_data.get("role", "assistant")),
                content=msg_data.get("content"),
            )
            choices.append(
                Choice(
                    index=c.get("index", 0),
                    message=message,
                    finish_reason=c.get("finish_reason"),
                )
            )
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            created=data.get("created"),
            raw=data,
        )

    def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
        if line == "[DONE]":
            return None
        import json

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        choices = []
        for c in data.get("choices", []):
            delta_data = c.get("delta", {})
            role_str = delta_data.get("role")
            delta = Message(
                role=Role(role_str) if role_str else Role.ASSISTANT,
                content=delta_data.get("content"),
            )
            choices.append(
                Choice(
                    index=c.get("index", 0),
                    delta=delta,
                    finish_reason=c.get("finish_reason"),
                )
            )
        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
        )

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_chat_payload(request)
        payload["stream"] = False
        data = await self._request("POST", "/chat/completions", payload)
        return self._parse_response(data)

    async def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        payload = self._build_chat_payload(request)
        payload["stream"] = True
        async for line in self._stream_request("/chat/completions", payload):
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
                    owned_by=m.get("owned_by", ""),
                    created=m.get("created", 0),
                )
            )
        return models
