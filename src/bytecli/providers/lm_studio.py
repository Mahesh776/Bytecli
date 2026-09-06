from collections.abc import AsyncIterator
from typing import Any, Literal, NamedTuple

from bytecli.config.schema import parse_model_params
from bytecli.logging.logger import get_logger
from bytecli.providers.openai_compatible import OpenAICompatibleProvider
from bytecli.providers.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    Role,
    Usage,
)

logger = get_logger("bytecli.providers")

ModelFamily = Literal["gemma", "llama", "mistral", "phi", "qwen", "deepseek", "chatml", "unknown"]


class PromptTemplate(NamedTuple):
    user_open: str
    user_close: str
    assistant_open: str
    assistant_close: str
    system_open: str
    system_close: str
    prefix: str
    suffix: str
    supports_system: bool = True


TEMPLATES: dict[ModelFamily, PromptTemplate] = {
    "gemma": PromptTemplate(
        user_open="<start_of_turn>user\n",
        user_close="<end_of_turn>\n",
        assistant_open="<start_of_turn>model\n",
        assistant_close="<end_of_turn>\n",
        system_open="",
        system_close="",
        prefix="",
        suffix="<start_of_turn>model\n",
        supports_system=False,
    ),
    "llama": PromptTemplate(
        user_open="<|start_header_id|>user<|end_header_id|>\n\n",
        user_close="<|eot_id|>\n",
        assistant_open="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_close="<|eot_id|>\n",
        system_open="<|start_header_id|>system<|end_header_id|>\n\n",
        system_close="<|eot_id|>\n",
        prefix="<|begin_of_text|>",
        suffix="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "mistral": PromptTemplate(
        user_open="[INST] ",
        user_close=" [/INST]",
        assistant_open="",
        assistant_close="</s>",
        system_open="",
        system_close="",
        prefix="<s>",
        suffix="",
        supports_system=False,
    ),
    "phi": PromptTemplate(
        user_open="<|user|>\n",
        user_close="<|end|>\n",
        assistant_open="<|assistant|>\n",
        assistant_close="<|end|>\n",
        system_open="<|system|>\n",
        system_close="<|end|>\n",
        prefix="",
        suffix="<|assistant|>\n",
    ),
    "qwen": PromptTemplate(
        user_open="<|im_start|>user\n",
        user_close="<|im_end|>\n",
        assistant_open="<|im_start|>assistant\n",
        assistant_close="<|im_end|>\n",
        system_open="<|im_start|>system\n",
        system_close="<|im_end|>\n",
        prefix="",
        suffix="<|im_start|>assistant\n",
    ),
    "deepseek": PromptTemplate(
        user_open="<｜User｜>",  # noqa: RUF001
        user_close="",
        assistant_open="<｜Assistant｜>",  # noqa: RUF001
        assistant_close="",
        system_open="",
        system_close="",
        prefix="<s>",
        suffix="<｜Assistant｜>",  # noqa: RUF001
        supports_system=False,
    ),
    "chatml": PromptTemplate(
        user_open="<|im_start|>user\n",
        user_close="<|im_end|>\n",
        assistant_open="<|im_start|>assistant\n",
        assistant_close="<|im_end|>\n",
        system_open="<|im_start|>system\n",
        system_close="<|im_end|>\n",
        prefix="",
        suffix="<|im_start|>assistant\n",
    ),
}

FALLBACK_TEMPLATE = PromptTemplate(
    user_open="User: ",
    user_close="\n",
    assistant_open="Assistant: ",
    assistant_close="\n",
    system_open="System: ",
    system_close="\n",
    prefix="",
    suffix="Assistant: ",
)


def _detect_model_family(model: str) -> ModelFamily:
    name = model.lower()
    if "gemma" in name:
        return "gemma"
    if "llama" in name or "llm" in name:
        return "llama"
    if "mistral" in name or "mixtral" in name:
        return "mistral"
    if "phi" in name:
        return "phi"
    if "qwen" in name:
        return "qwen"
    if "deepseek" in name:
        return "deepseek"
    if "chatml" in name:
        return "chatml"
    return "unknown"


def _format_prompt(messages: list[Message], template: PromptTemplate) -> str:
    parts: list[str] = []
    if template.prefix:
        parts.append(template.prefix)

    pending_system: list[str] = []

    for msg in messages:
        role = msg.role.value
        content = msg.content or ""
        if role == "system":
            if template.supports_system:
                parts.append(template.system_open + content + template.system_close)
            else:
                pending_system.append(content)
            continue

        if role == "user" and pending_system:
            content = "\n".join(pending_system) + "\n" + content
            pending_system.clear()

        if role == "user":
            parts.append(template.user_open + content + template.user_close)
        elif role == "assistant":
            parts.append(template.assistant_open + content + template.assistant_close)
        elif role == "tool":
            name = msg.name or "tool"
            parts.append(template.user_open + f"[Tool result from {name}]:\n{content}" + template.user_close)

    if pending_system and not any(m.role.value == "user" for m in messages):
        parts.append(template.user_open + "\n".join(pending_system) + template.user_close)

    parts.append(template.suffix)
    return "".join(parts)


class LMStudioProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._template_cache: dict[str, PromptTemplate] = {}
        self._resolved_model: str | None = None

    def _pick_best_model(self, models: list[str]) -> str | None:
        scored: list[tuple[float, int, str]] = []
        for index, model in enumerate(models):
            name = model.lower()
            if "embed" in name:
                continue
            score = 0.0
            if "coder" in name:
                score += 3.0
            if "instruct" in name:
                score += 1.0
            if "chat" in name:
                score += 0.5
            if name.endswith("-it"):
                score += 1.0
            params = parse_model_params(model)
            if params is not None:
                score += min(params, 20.0) / 10.0
            scored.append((score, -index, model))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    async def _resolve_real_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        try:
            data: dict[str, Any] = await self._request("GET", "/models")
            raw_models: Any = data.get("data", [])
            models = [str(m.get("id", "")) for m in raw_models if isinstance(m, dict) and m.get("id")]
            best = self._pick_best_model(models)
            if best:
                self._resolved_model = best
                logger.info("Resolved LM Studio model name: {}", best)
                return best
        except Exception:
            logger.warning("Could not resolve model name from LM Studio")
        return "unknown"

    async def resolve_model_name(self) -> str | None:
        resolved = await self._resolve_real_model()
        return resolved if resolved != "unknown" else None

    def _get_template(self, model: str) -> PromptTemplate:
        if model in self._template_cache:
            return self._template_cache[model]
        family = _detect_model_family(model)
        template = TEMPLATES.get(family, FALLBACK_TEMPLATE)
        self._template_cache[model] = template
        logger.debug(
            "Detected model family '{}' for '{}', using template prefix={!r}",
            family,
            model,
            template.prefix[:20],
        )
        return template

    def _messages_to_prompt(self, messages: list[Message], model: str = "") -> str:
        template = self._get_template(model)
        return _format_prompt(messages, template)

    async def _get_model_name(self, request_model: str) -> str:
        if request_model in ("local-model", "unknown", ""):
            resolved = await self._resolve_real_model()
            return resolved if resolved != "unknown" else request_model
        return request_model

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        model = await self._get_model_name(request.model)
        resolved = CompletionRequest(
            model=model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop,
            stream=False,
            extra=request.extra,
        )
        return await super().chat(resolved)

    async def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        model = await self._get_model_name(request.model)
        resolved = CompletionRequest(
            model=model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop,
            stream=True,
            extra=request.extra,
        )
        async for chunk in super().chat_stream(resolved):
            yield chunk
