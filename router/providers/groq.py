"""
router/providers/groq.py — Groq provider (OpenAI-compatible API).

Groq uses the OpenAI SDK pointed at https://api.groq.com/openai/v1.
System prompt is passed as the first message with role="system".
"""
from __future__ import annotations

import time

import openai

from .base import BaseProvider, ProviderResponse

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(BaseProvider):
    """Calls Groq's OpenAI-compatible inference API."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
        )

    @property
    def provider_name(self) -> str:
        return "groq"

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
