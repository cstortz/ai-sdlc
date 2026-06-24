"""
router/providers/together.py — Together AI provider (OpenAI-compatible API).

Together AI uses the OpenAI SDK pointed at https://api.together.xyz/v1.
"""
from __future__ import annotations

import time

import openai

from .base import BaseProvider, ProviderResponse

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class TogetherProvider(BaseProvider):
    """Calls Together AI's OpenAI-compatible inference API."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=TOGETHER_BASE_URL,
        )

    @property
    def provider_name(self) -> str:
        return "together"

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

        full_messages = (
            [{"role": "system", "content": system}] + messages
            if system
            else list(messages)
        )

        response = await self._client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        duration_ms = int((time.monotonic() - t0) * 1000)
        content = response.choices[0].message.content or ""
        usage = response.usage

        return ProviderResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            duration_ms=duration_ms,
            raw=response,
        )
