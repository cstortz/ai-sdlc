"""
router/providers/base.py — Abstract base class for all LLM providers.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """Normalized response returned by every provider."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    duration_ms: int
    raw: object = None  # Provider-native response object for debugging


class BaseProvider(ABC):
    """
    All providers implement this interface.

    Providers are stateless — they hold only the API key and base config.
    Retry logic lives here in the base class so concrete providers stay thin.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        retry_attempts: int = 3,
    ) -> ProviderResponse:
        """
        Call the provider with exponential-backoff retry.
        Raises ProviderError after all attempts are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, retry_attempts + 1):
            try:
                return await self._call(
                    model=model,
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < retry_attempts:
                    wait = 2 ** attempt  # 2s, 4s, 8s …
                    logger.warning(
                        "%s attempt %d/%d failed (%s). Retrying in %ds.",
                        self.__class__.__name__, attempt, retry_attempts, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "%s failed after %d attempts: %s",
                        self.__class__.__name__, retry_attempts, exc,
                    )

        from router.exceptions import ProviderError
        raise ProviderError(
            provider=self.provider_name,
            model=model,
            message=str(last_exc),
        )

    @abstractmethod
    async def _call(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> ProviderResponse:
        """Provider-specific implementation. Must be overridden."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier, e.g. 'anthropic', 'groq', 'together'."""
