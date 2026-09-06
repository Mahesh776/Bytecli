from unittest.mock import AsyncMock, patch

import pytest

from bytecli.providers.anthropic import AnthropicProvider
from bytecli.providers.base_http import BaseHTTPProvider
from bytecli.providers.factory import ProviderFactory
from bytecli.providers.gemini import GeminiProvider
from bytecli.providers.lm_studio import LMStudioProvider
from bytecli.providers.openai import OpenAIProvider
from bytecli.providers.openai_compatible import OpenAICompatibleProvider
from bytecli.providers.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    ModelInfo,
    Role,
    ToolCall,
    Usage,
)


class TestTypes:
    def test_role_values(self) -> None:
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_message_creation(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="call_1", function={"name": "get_weather"})
        msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[tc])
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].id == "call_1"

    def test_completion_request_defaults(self) -> None:
        req = CompletionRequest(
            model="test-model",
            messages=[Message(role=Role.USER, content="Hi")],
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 4096
        assert req.top_p == 0.95
        assert req.stream is False

    def test_completion_response(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="Hello")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        resp = CompletionResponse(id="resp_1", model="test", choices=[choice])
        assert resp.id == "resp_1"
        assert resp.choices[0].message is not None
        assert resp.choices[0].message.content == "Hello"

    def test_usage(self) -> None:
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30

    def test_model_info(self) -> None:
        info = ModelInfo(id="gpt-4", owned_by="openai", created=12345)
        assert info.id == "gpt-4"
        assert info.owned_by == "openai"


class TestOpenAICompatibleProvider:
    @pytest.fixture
    def provider(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            base_url="http://test.local/v1",
            api_key="test-key",
            timeout=10.0,
            max_retries=1,
        )

    def test_build_headers(self, provider: OpenAICompatibleProvider) -> None:
        headers = provider._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_no_key(self) -> None:
        p = OpenAICompatibleProvider(base_url="http://test.local/v1")
        headers = p._build_headers()
        assert "Authorization" not in headers

    def test_build_chat_payload(self, provider: OpenAICompatibleProvider) -> None:
        messages = [Message(role=Role.USER, content="Hello")]
        req = CompletionRequest(model="test-model", messages=messages)
        payload = provider._build_chat_payload(req)
        assert payload["model"] == "test-model"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["stream"] is False
        assert payload["temperature"] == 0.7

    def test_build_chat_payload_with_stop(self, provider: OpenAICompatibleProvider) -> None:
        req = CompletionRequest(
            model="test",
            messages=[Message(role=Role.USER, content="Hi")],
            stop=["\n", "stop"],
        )
        payload = provider._build_chat_payload(req)
        assert payload["stop"] == ["\n", "stop"]

    def test_parse_response(self, provider: OpenAICompatibleProvider) -> None:
        data = {
            "id": "chat_1",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "created": 1234567890,
        }
        resp = provider._parse_response(data)
        assert resp.id == "chat_1"
        assert resp.model == "gpt-4"
        assert resp.choices[0].message is not None
        assert resp.choices[0].message.content == "Hello!"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 15

    def test_parse_stream_chunk(self, provider: OpenAICompatibleProvider) -> None:
        chunk_data = (
            '{"id":"chunk_1","model":"gpt-4","choices":['
            '{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}'
        )
        resp = provider._parse_stream_chunk(chunk_data)
        assert resp is not None
        assert resp.id == "chunk_1"
        assert resp.choices[0].delta is not None
        assert resp.choices[0].delta.content == "Hello"

    def test_parse_stream_chunk_done(self, provider: OpenAICompatibleProvider) -> None:
        resp = provider._parse_stream_chunk("[DONE]")
        assert resp is None

    def test_parse_stream_chunk_invalid(self, provider: OpenAICompatibleProvider) -> None:
        resp = provider._parse_stream_chunk("not json")
        assert resp is None

    def test_messages_to_dicts(self, provider: OpenAICompatibleProvider) -> None:
        messages = [
            Message(role=Role.SYSTEM, content="Be helpful"),
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello!"),
        ]
        result = provider._messages_to_dicts(messages)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Hi"

    def test_messages_to_dicts_with_tool_calls(self, provider: OpenAICompatibleProvider) -> None:
        tc = ToolCall(id="call_1", function={"name": "get_weather", "arguments": "{}"})
        messages = [
            Message(role=Role.ASSISTANT, tool_calls=[tc]),
            Message(role=Role.TOOL, tool_call_id="call_1", content='{"temp": 72}'),
        ]
        result = provider._messages_to_dicts(messages)
        assert result[0]["tool_calls"][0]["id"] == "call_1"
        assert result[1]["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_validate_connection_success(self, provider: OpenAICompatibleProvider) -> None:
        with patch.object(provider, "_request", new=AsyncMock(return_value={"data": []})):
            result = await provider.validate_connection()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self, provider: OpenAICompatibleProvider) -> None:
        from bytecli.core.errors import ProviderError
        with patch.object(provider, "_request", new=AsyncMock(side_effect=ProviderError("fail"))):
            result = await provider.validate_connection()
            assert result is False


class TestProviderFactory:
    def test_create_lm_studio(self) -> None:
        from bytecli.config.schema import ProviderEndpointConfig, ProviderType
        config = ProviderEndpointConfig(base_url="http://localhost:1234/v1")
        provider = ProviderFactory.create(ProviderType.LM_STUDIO, config)
        assert isinstance(provider, LMStudioProvider)
        assert provider.base_url == "http://localhost:1234/v1"

    def test_create_openai(self) -> None:
        from bytecli.config.schema import ProviderEndpointConfig, ProviderType
        config = ProviderEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-xxx")
        provider = ProviderFactory.create(ProviderType.OPENAI, config)
        assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic(self) -> None:
        from bytecli.config.schema import ProviderEndpointConfig, ProviderType
        config = ProviderEndpointConfig(base_url="https://api.anthropic.com/v1", api_key="sk-ant-xxx")
        provider = ProviderFactory.create(ProviderType.ANTHROPIC, config)
        assert isinstance(provider, AnthropicProvider)

    def test_create_gemini(self) -> None:
        from bytecli.config.schema import ProviderEndpointConfig, ProviderType
        config = ProviderEndpointConfig(base_url="https://generativelanguage.googleapis.com/v1beta", api_key="AI-xxx")
        provider = ProviderFactory.create(ProviderType.GEMINI, config)
        assert isinstance(provider, GeminiProvider)

    def test_create_all_providers(self) -> None:
        from bytecli.config.schema import ProviderEndpointConfig, ProviderType
        config = ProviderEndpointConfig()
        for pt in ProviderType:
            provider = ProviderFactory.create(pt, config)
            assert provider is not None

    def test_create_invalid_type(self) -> None:
        from bytecli.config.schema import ProviderEndpointConfig
        config = ProviderEndpointConfig()
        with pytest.raises(ValueError, match="Unknown provider type"):
            ProviderFactory.create("invalid", config)  # type: ignore[arg-type]


class TestBaseHTTPProvider:
    @pytest.fixture
    def provider(self) -> BaseHTTPProvider:
        return OpenAICompatibleProvider(
            base_url="http://test.local/v1",
            api_key="test-key",
            timeout=10.0,
            max_retries=1,
        )

    @pytest.mark.asyncio
    async def test_close(self, provider: BaseHTTPProvider) -> None:
        provider._get_client()
        assert provider._client is not None
        await provider.close()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_validate_connection_http_error(self, provider: BaseHTTPProvider) -> None:
        result = await provider.validate_connection()
        assert result is False  # No server running at test.local

    @pytest.mark.asyncio
    async def test_provider_error_on_connect_error(self, provider: BaseHTTPProvider) -> None:
        from bytecli.core.errors import ProviderConnectionError
        with pytest.raises(ProviderConnectionError):
            await provider._request("GET", "/models")


class TestLMStudioDetectFamily:
    def test_detect_gemma(self) -> None:
        from bytecli.providers.lm_studio import _detect_model_family
        assert _detect_model_family("gemma-3-270m") == "gemma"
        assert _detect_model_family("google/gemma-2b") == "gemma"

    def test_detect_llama(self) -> None:
        from bytecli.providers.lm_studio import _detect_model_family
        assert _detect_model_family("llama-3.1-8b") == "llama"
        assert _detect_model_family("meta-llama-3") == "llama"

    def test_detect_mistral(self) -> None:
        from bytecli.providers.lm_studio import _detect_model_family
        assert _detect_model_family("mistral-7b") == "mistral"
        assert _detect_model_family("mixtral-8x7b") == "mistral"

    def test_detect_phi(self) -> None:
        from bytecli.providers.lm_studio import _detect_model_family
        assert _detect_model_family("phi-3-mini") == "phi"

    def test_detect_qwen(self) -> None:
        from bytecli.providers.lm_studio import _detect_model_family
        assert _detect_model_family("qwen2.5-7b") == "qwen"

    def test_detect_deepseek(self) -> None:
        from bytecli.providers.lm_studio import _detect_model_family
        assert _detect_model_family("deepseek-coder") == "deepseek"

    def test_detect_unknown(self) -> None:
        from bytecli.providers.lm_studio import _detect_model_family
        assert _detect_model_family("some-unknown-model") == "unknown"


class TestLMStudioFormatPrompt:
    def test_format_gemma(self) -> None:
        from bytecli.providers.lm_studio import _format_prompt, TEMPLATES
        template = TEMPLATES["gemma"]
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
        prompt = _format_prompt(messages, template)
        assert prompt.startswith("<start_of_turn>user")
        assert "You are helpful." in prompt
        assert "Hello" in prompt
        assert "Hi there!" in prompt
        assert prompt.endswith("<start_of_turn>model\n")

    def test_format_gemma_simple(self) -> None:
        from bytecli.providers.lm_studio import _format_prompt, TEMPLATES
        template = TEMPLATES["gemma"]
        messages = [
            Message(role=Role.USER, content="Hi"),
        ]
        prompt = _format_prompt(messages, template)
        assert "<start_of_turn>user\nHi<end_of_turn>" in prompt
        assert prompt.rstrip().endswith("<start_of_turn>model")

    def test_format_llama(self) -> None:
        from bytecli.providers.lm_studio import _format_prompt, TEMPLATES
        template = TEMPLATES["llama"]
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="World"),
        ]
        prompt = _format_prompt(messages, template)
        assert prompt.startswith("<|begin_of_text|>")
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "Hello" in prompt
        assert "World" in prompt
        assert prompt.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")

    def test_format_fallback(self) -> None:
        from bytecli.providers.lm_studio import FALLBACK_TEMPLATE, _format_prompt
        messages = [
            Message(role=Role.USER, content="Hi"),
        ]
        prompt = _format_prompt(messages, FALLBACK_TEMPLATE)
        assert prompt == "User: Hi\nAssistant: "

    def test_format_with_tool_result(self) -> None:
        from bytecli.providers.lm_studio import _format_prompt, TEMPLATES
        template = TEMPLATES["gemma"]
        messages = [
            Message(role=Role.USER, content="Run tool"),
            Message(role=Role.ASSISTANT, content=None, tool_calls=[]),
            Message(role=Role.TOOL, content="42", name="calculator"),
        ]
        prompt = _format_prompt(messages, template)
        assert "Tool result from calculator" in prompt
        assert "42" in prompt
        assert prompt.rstrip().endswith("<start_of_turn>model")


class TestLMStudioProvider:
    def test_messages_to_prompt_detects_model(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        messages = [Message(role=Role.USER, content="Hi")]
        # Unknown model → fallback template
        prompt = provider._messages_to_prompt(messages, model="unknown-model")
        assert prompt == "User: Hi\nAssistant: "
        # Gemma model → gemma template
        prompt2 = provider._messages_to_prompt(messages, model="gemma-3-270m")
        assert "<start_of_turn>user" in prompt2
        assert prompt2.rstrip().endswith("<start_of_turn>model")
        # Template is cached
        assert "unknown-model" in provider._template_cache
        assert "gemma-3-270m" in provider._template_cache


class TestLMStudioModelPicker:
    def test_pick_prefers_coder_instruct(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        models = [
            "gemma-3-270m-it",
            "qwen2.5-coder-1.5b-instruct",
            "google.gemma-3-1b-it",
            "huggingfacetb_smollm3-3b",
        ]
        assert provider._pick_best_model(models) == "qwen2.5-coder-1.5b-instruct"

    def test_pick_excludes_embedding_models(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        models = [
            "text-embedding-nomic-embed-text-v1.5",
            "gemma-3-270m-it",
        ]
        assert provider._pick_best_model(models) == "gemma-3-270m-it"

    def test_pick_returns_none_when_only_embeddings(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        assert provider._pick_best_model(["text-embedding-nomic-embed-text-v1.5"]) is None

    def test_pick_empty_list(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        assert provider._pick_best_model([]) is None

    def test_pick_keeps_first_on_tie(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        models = [
            "qwen2.5-coder-0.5b-instruct",
            "unsloth/qwen2.5-coder-0.5b-instruct",
        ]
        assert provider._pick_best_model(models) == models[0]

    @pytest.mark.asyncio
    async def test_resolve_real_model_picks_best(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        provider._request = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "data": [
                    {"id": "gemma-3-270m-it"},
                    {"id": "qwen2.5-coder-1.5b-instruct"},
                    {"id": "text-embedding-nomic-embed-text-v1.5"},
                ]
            }
        )
        assert await provider.resolve_model_name() == "qwen2.5-coder-1.5b-instruct"

    @pytest.mark.asyncio
    async def test_resolve_real_model_falls_back_to_unknown(self) -> None:
        from bytecli.providers.lm_studio import LMStudioProvider
        provider = LMStudioProvider()
        provider._request = AsyncMock(side_effect=RuntimeError("down"))  # type: ignore[method-assign]
        assert await provider.resolve_model_name() is None
