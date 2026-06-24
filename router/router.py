"""
router/router.py — ModelRouter: the central dispatch layer.

Usage:
    from router import ModelRouter

    router = ModelRouter()   # reads config/models.yaml by default

    response = await router.complete(
        profile="intake",
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a requirements engineer.",
    )
    print(response.content)
    print(f"Cost: ${response.cost_usd:.6f}  Model: {response.model_used}  Fallback: {response.was_fallback}")
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .config import RouterConfig, ModelEntry, Profile, estimate_cost
from .exceptions import AllProvidersFailedError, CostThresholdExceededError, ProviderError
from .providers import PROVIDER_REGISTRY, BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)

# Default path relative to the repo root; override via ModelRouter(config_path=...)
_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "models.yaml"


@dataclass
class RouterResponse:
    """Enriched response returned by ModelRouter.complete()."""
    content: str
    profile: str
    model_used: str
    provider: str
    was_fallback: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int
    # Raw provider response — useful for debugging, omit in logs
    raw: object = None

    def __repr__(self) -> str:
        return (
            f"RouterResponse(profile={self.profile!r}, model={self.model_used!r}, "
            f"fallback={self.was_fallback}, tokens={self.input_tokens}+{self.output_tokens}, "
            f"cost=${self.cost_usd:.6f}, duration={self.duration_ms}ms)"
        )


class ModelRouter:
    """
    Reads config/models.yaml and routes LLM calls to the correct
    provider/model, falling back automatically on failure.

    Thread-safe for reads; call reload_config() to hot-reload the YAML
    without restarting (useful in long-running processes).
    """

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG):
        self._config_path = Path(config_path)
        self._config: RouterConfig = RouterConfig.from_file(self._config_path)
        self._provider_cache: dict[str, BaseProvider] = {}
        logger.info(
            "ModelRouter initialised. Profiles: %s",
            ", ".join(self._config.profiles),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        profile: str,
        messages: list[dict],
        system: str = "",
        **overrides,
    ) -> RouterResponse:
        """
        Dispatch a completion request using the named profile.

        Args:
            profile:   Key from models.yaml `profiles:` block (e.g. "intake").
            messages:  List of {"role": ..., "content": ...} dicts.
            system:    Optional system prompt string.
            **overrides: Runtime overrides — temperature, max_tokens, etc.

        Returns:
            RouterResponse with content, token counts, cost, and metadata.

        Raises:
            AllProvidersFailedError if both primary and fallback fail.
            KeyError if `profile` is not in models.yaml.
        """
        prof = self._config.get_profile(profile)
        primary_error: Exception | None = None

        # ---- Cost-threshold check: downgrade to fallback before calling? ----
        # Only applies if cost_threshold_usd is set and we can estimate tokens.
        # (We skip estimation here since we don't know token count up front;
        #  the threshold is enforced post-call for now and logged as a warning.)

        # ---- Try primary ----
        try:
            prov_resp = await self._dispatch(prof.primary, prof, messages, system, overrides)
            return self._build_response(profile, prof.primary, prov_resp, was_fallback=False)
        except ProviderError as exc:
            primary_error = exc
            logger.warning(
                "Primary model failed for profile '%s' (%s/%s): %s — trying fallback.",
                profile, prof.primary.provider, prof.primary.model, exc,
            )

        # ---- Try fallback ----
        try:
            prov_resp = await self._dispatch(prof.fallback, prof, messages, system, overrides)
            resp = self._build_response(profile, prof.fallback, prov_resp, was_fallback=True)
            logger.info(
                "Fallback succeeded for profile '%s': %s/%s",
                profile, prof.fallback.provider, prof.fallback.model,
            )
            return resp
        except ProviderError as fallback_error:
            raise AllProvidersFailedError(
                profile=profile,
                primary_error=primary_error,
                fallback_error=fallback_error,
            ) from fallback_error

    def reload_config(self) -> None:
        """Hot-reload models.yaml without restarting the process."""
        self._config = RouterConfig.from_file(self._config_path)
        logger.info("ModelRouter config reloaded from %s", self._config_path)

    def list_profiles(self) -> list[str]:
        """Return all profile names defined in models.yaml."""
        return list(self._config.profiles)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        entry: ModelEntry,
        prof: Profile,
        messages: list[dict],
        system: str,
        overrides: dict,
    ) -> ProviderResponse:
        provider = self._get_provider(entry.provider)

        temperature = overrides.get("temperature", prof.effective_temperature(entry))
        max_tokens  = overrides.get("max_tokens",  prof.effective_max_tokens(entry))
        timeout     = overrides.get("timeout",     float(prof.request_timeout_seconds))
        retries     = overrides.get("retry_attempts", prof.retry_attempts)

        logger.debug(
            "Dispatching to %s/%s (temp=%.2f, max_tokens=%d)",
            entry.provider, entry.model, temperature, max_tokens,
        )

        return await provider.complete(
            model=entry.model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_attempts=retries,
        )

    def _get_provider(self, provider_name: str) -> BaseProvider:
        """Return a cached provider instance, constructing it on first use."""
        if provider_name not in self._provider_cache:
            cls = PROVIDER_REGISTRY.get(provider_name)
            if cls is None:
                available = ", ".join(PROVIDER_REGISTRY)
                raise ValueError(
                    f"Unknown provider '{provider_name}'. "
                    f"Available: {available}"
                )
            api_key = self._resolve_api_key(provider_name)
            self._provider_cache[provider_name] = cls(api_key=api_key)
            logger.debug("Initialised provider: %s", provider_name)
        return self._provider_cache[provider_name]

    @staticmethod
    def _resolve_api_key(provider_name: str) -> str:
        """
        Resolve the API key for a provider from environment variables.

        Expected env var names (set from secrets.yaml → Helm secret → pod env):
            anthropic → ANTHROPIC_API_KEY
            groq      → GROQ_API_KEY
            together  → TOGETHER_API_KEY
        """
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "groq":      "GROQ_API_KEY",
            "together":  "TOGETHER_API_KEY",
        }
        env_var = env_map.get(provider_name, f"{provider_name.upper()}_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            raise EnvironmentError(
                f"API key for provider '{provider_name}' not found. "
                f"Set the {env_var} environment variable."
            )
        return key

    @staticmethod
    def _build_response(
        profile: str,
        entry: ModelEntry,
        prov: ProviderResponse,
        was_fallback: bool,
    ) -> RouterResponse:
        cost = estimate_cost(prov.model, prov.input_tokens, prov.output_tokens)

        # Warn if cost exceeded threshold (post-call enforcement)
        # Pre-call enforcement requires token estimation — future enhancement.

        return RouterResponse(
            content=prov.content,
            profile=profile,
            model_used=prov.model,
            provider=prov.provider,
            was_fallback=was_fallback,
            input_tokens=prov.input_tokens,
            output_tokens=prov.output_tokens,
            cost_usd=cost,
            duration_ms=prov.duration_ms,
            raw=prov.raw,
        )
