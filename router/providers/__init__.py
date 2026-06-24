"""
router/providers — LLM provider implementations.

Registry maps provider name strings (as used in models.yaml) to classes.
"""
from .anthropic import AnthropicProvider
from .base import BaseProvider, ProviderResponse
from .groq import GroqProvider
from .together import TogetherProvider

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "together": TogetherProvider,
}

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "AnthropicProvider",
    "GroqProvider",
    "TogetherProvider",
    "PROVIDER_REGISTRY",
]
