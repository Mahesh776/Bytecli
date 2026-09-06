from typing import ClassVar

from bytecli.config.schema import ProviderEndpointConfig, ProviderType
from bytecli.providers.anthropic import AnthropicProvider
from bytecli.providers.base import Provider
from bytecli.providers.deepseek import DeepSeekProvider
from bytecli.providers.gemini import GeminiProvider
from bytecli.providers.lm_studio import LMStudioProvider
from bytecli.providers.mistral import MistralProvider
from bytecli.providers.ollama import OllamaProvider
from bytecli.providers.openai import OpenAIProvider
from bytecli.providers.openrouter import OpenRouterProvider
from bytecli.providers.vllm import VLLMProvider


class ProviderFactory:
    _registry: ClassVar[dict[ProviderType, type[Provider]]] = {
        ProviderType.LM_STUDIO: LMStudioProvider,
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.VLLM: VLLMProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.OPENROUTER: OpenRouterProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.GEMINI: GeminiProvider,
        ProviderType.MISTRAL: MistralProvider,
        ProviderType.DEEPSEEK: DeepSeekProvider,
    }

    @classmethod
    def create(cls, provider_type: ProviderType, config: ProviderEndpointConfig) -> Provider:
        provider_cls = cls._registry.get(provider_type)
        if provider_cls is None:
            raise ValueError(f"Unknown provider type: {provider_type}")
        return provider_cls(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
