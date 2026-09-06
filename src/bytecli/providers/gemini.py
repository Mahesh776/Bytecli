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


class GeminiProvider(BaseHTTPProvider):
    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(base_url, api_key, timeout, max_retries)

    def _build_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_instruction = None

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_instruction = {"parts": [{"text": msg.content or ""}]}
            else:
                role = "model" if msg.role == Role.ASSISTANT else "user"
                contents.append(
                    {
                        "role": role,
                        "parts": [{"text": msg.content or ""}],
                    }
                )

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": request.top_p,
            },
        }
        if system_instruction is not None:
            payload["system_instruction"] = system_instruction
        if request.stop:
            payload["generationConfig"]["stopSequences"] = request.stop
        payload.update(request.extra)
        return payload

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        candidates = data.get("candidates", [])
        choices: list[Choice] = []

        for c in candidates:
            content = c.get("content", {})
            parts = content.get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            message = Message(role=Role.ASSISTANT, content=text or None)
            finish = c.get("finishReason", "").lower() if c.get("finishReason") else None
            choices.append(Choice(index=c.get("index", 0), message=message, finish_reason=finish))

        usage_data = data.get("usageMetadata")
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
                total_tokens=usage_data.get("totalTokenCount", 0),
            )

        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices or [Choice(index=0, message=Message(role=Role.ASSISTANT, content=None))],
            usage=usage,
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

        candidates = data.get("candidates", [])
        choices: list[Choice] = []
        for c in candidates:
            content = c.get("content", {})
            parts = content.get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if text:
                message = Message(role=Role.ASSISTANT, content=text)
                choices.append(Choice(index=c.get("index", 0), delta=message))

        if not choices:
            return None

        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
        )

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_chat_payload(request)
        data = await self._request("POST", f"/models/{request.model}:generateContent?key={self.api_key or ''}", payload)
        return self._parse_response(data)

    async def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        payload = self._build_chat_payload(request)
        path = f"/models/{request.model}:streamGenerateContent?alt=sse&key={self.api_key or ''}"
        async for line in self._stream_request(path, payload):
            chunk = self._parse_stream_chunk(line)
            if chunk is not None:
                yield chunk

    async def list_models(self) -> list[ModelInfo]:
        data = await self._request("GET", f"/models?key={self.api_key or ''}")
        models: list[ModelInfo] = []
        for m in data.get("models", []):
            models.append(
                ModelInfo(
                    id=m.get("name", "").replace("models/", ""),
                    owned_by="google",
                    created=0,
                )
            )
        return models

    async def validate_connection(self) -> bool:
        try:
            await self._request("GET", f"/models?key={self.api_key or ''}")
            return True
        except ProviderError:
            return False
