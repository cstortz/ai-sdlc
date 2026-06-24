"""
router/providers/anthropic.py — Anthropic provider (claude-* models).

Uses the official `anthropic` async SDK.
"""
from __future__ import annotations

import time

import anthropic

from .base import BaseProvider, ProviderResponse


class AnthropicProvider(BaseProvider):
    """Calls Anthropic's Messages API."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

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
        t0 = time.monotonic()

        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)

        duration_ms = int((time.monotonic() - t0) * 1000)
        content = response.content[0].text if response.content else ""

        return ProviderResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            duration_ms=duration_ms,
            raw=response,
        )
