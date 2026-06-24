"""
router/config.py — Dataclasses for parsing config/models.yaml.

Loaded once at ModelRouter init; re-loadable via router.reload_config().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Pricing table — cost per 1M tokens (input / output).
# Update these when provider pricing changes.
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-8":          {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":        {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":{"input":  0.25, "output":  1.25},
    # Groq (open-source inference)
    "llama-3.3-70b-versatile":  {"input":  0.59, "output":  0.79},
    "mixtral-8x7b-32768":       {"input":  0.24, "output":  0.24},
    # Together AI
    "meta-llama/Llama-3.3-70B-Instruct": {"input": 0.18, "output": 0.18},
    "mistral-large-latest":     {"input":  2.00, "output":  6.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD; returns 0.0 if model is not in PRICING."""
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """A single model + provider pair (primary or fallback)."""
    model: str
    provider: str
    # Optional overrides — None means "use profile/global default"
    temperature: float | None = None
    max_tokens: int | None = None
    cost_threshold_usd: float | None = None


@dataclass
class Profile:
    """One entry from models.yaml `profiles:` block."""
    name: str
    primary: ModelEntry
    fallback: ModelEntry
    # Resolved from profile-level keys, then defaults
    temperature: float = 0.2
    max_tokens: int = 4096
    request_timeout_seconds: int = 120
    retry_attempts: int = 3
    cost_threshold_usd: float | None = None

    def effective_temperature(self, entry: ModelEntry) -> float:
        return entry.temperature if entry.temperature is not None else self.temperature

    def effective_max_tokens(self, entry: ModelEntry) -> int:
        return entry.max_tokens if entry.max_tokens is not None else self.max_tokens

    def effective_cost_threshold(self, entry: ModelEntry) -> float | None:
        return entry.cost_threshold_usd if entry.cost_threshold_usd is not None \
               else self.cost_threshold_usd


@dataclass
class RouterConfig:
    """Parsed representation of the full models.yaml file."""
    profiles: dict[str, Profile] = field(default_factory=dict)
    # Raw defaults block (applied during parsing)
    _defaults: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "RouterConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "RouterConfig":
        defaults = raw.get("defaults") or {}   # `or {}` handles defaults: with all keys commented out (parses as None)
        profiles: dict[str, Profile] = {}

        for name, pdata in raw.get("profiles", {}).items():
            primary = cls._parse_entry(pdata["primary"])
            fallback = cls._parse_entry(pdata["fallback"])

            profiles[name] = Profile(
                name=name,
                primary=primary,
                fallback=fallback,
                temperature=pdata.get(
                    "temperature",
                    defaults.get("temperature", 0.2)
                ),
                max_tokens=pdata.get(
                    "max_tokens",
                    defaults.get("max_tokens", 4096)
                ),
                request_timeout_seconds=pdata.get(
                    "request_timeout_seconds",
                    defaults.get("request_timeout_seconds", 120)
                ),
                retry_attempts=pdata.get(
                    "retry_attempts",
                    defaults.get("retry_attempts", 3)
                ),
                cost_threshold_usd=pdata.get(
                    "cost_threshold_usd",
                    defaults.get("cost_threshold_usd")
                ),
            )

        return cls(profiles=profiles, _defaults=defaults)

    @staticmethod
    def _parse_entry(raw: dict) -> ModelEntry:
        return ModelEntry(
            model=raw["model"],
            provider=raw["provider"],
            temperature=raw.get("temperature"),
            max_tokens=raw.get("max_tokens"),
            cost_threshold_usd=raw.get("cost_threshold_usd"),
        )

    def get_profile(self, name: str) -> Profile:
        if name not in self.profiles:
            available = ", ".join(self.profiles)
            raise KeyError(
                f"Model profile '{name}' not found in models.yaml. "
                f"Available: {available}"
            )
        return self.profiles[name]
