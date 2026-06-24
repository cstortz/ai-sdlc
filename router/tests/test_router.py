"""
router/tests/test_router.py — Unit tests for the model router.

Run with:
    pip install pytest pytest-asyncio --break-system-packages
    pytest router/tests/ -v

These tests use mocks — no real API calls are made.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from router.config import RouterConfig, ModelEntry, Profile, estimate_cost, PRICING
from router.exceptions import AllProvidersFailedError, ProviderError
from router.providers.base import ProviderResponse
from router.router import ModelRouter, RouterResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODELS_YAML = Path(__file__).parent.parent.parent / "config" / "models.yaml"


@pytest.fixture
def config() -> RouterConfig:
    """Load the real models.yaml — validates the file parses correctly."""
    return RouterConfig.from_file(MODELS_YAML)


def _make_provider_response(model: str = "claude-sonnet-4-6", provider: str = "anthropic") -> ProviderResponse:
    return ProviderResponse(
        content="Test response content",
        model=model,
        provider=provider,
        input_tokens=100,
        output_tokens=50,
        duration_ms=250,
        raw=None,
    )


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------

class TestRouterConfig:

    def test_loads_real_models_yaml(self, config: RouterConfig):
        """models.yaml parses without error and contains expected profiles."""
        assert len(config.profiles) > 0

    def test_all_expected_profiles_present(self, config: RouterConfig):
        expected = {
            "intake", "architecture", "implementation",
            "testing", "security_scan", "code_review",
            "pr_description", "deploy_orchestration",
            "gate_evaluation", "monitor_triage", "incident_analysis",
        }
        assert expected.issubset(set(config.profiles))

    def test_profile_has_primary_and_fallback(self, config: RouterConfig):
        for name, prof in config.profiles.items():
            assert prof.primary.model, f"Profile '{name}' missing primary.model"
            assert prof.primary.provider, f"Profile '{name}' missing primary.provider"
            assert prof.fallback.model, f"Profile '{name}' missing fallback.model"
            assert prof.fallback.provider, f"Profile '{name}' missing fallback.provider"

    def test_get_profile_raises_on_unknown(self, config: RouterConfig):
        with pytest.raises(KeyError, match="not found"):
            config.get_profile("does_not_exist")

    def test_effective_temperature_uses_entry_override(self):
        entry = ModelEntry(model="x", provider="anthropic", temperature=0.9)
        prof = Profile(name="p", primary=entry, fallback=entry, temperature=0.1)
        assert prof.effective_temperature(entry) == 0.9

    def test_effective_temperature_falls_back_to_profile(self):
        entry = ModelEntry(model="x", provider="anthropic")  # no temp override
        prof = Profile(name="p", primary=entry, fallback=entry, temperature=0.3)
        assert prof.effective_temperature(entry) == 0.3


# ---------------------------------------------------------------------------
# Cost estimation tests
# ---------------------------------------------------------------------------

class TestCostEstimation:

    def test_known_model(self):
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(18.0)  # 3.00 + 15.00

    def test_unknown_model_returns_zero(self):
        assert estimate_cost("unknown-model-xyz", 1000, 1000) == 0.0

    def test_zero_tokens(self):
        assert estimate_cost("claude-opus-4-8", 0, 0) == 0.0

    def test_all_pricing_entries_have_input_output(self):
        for model, prices in PRICING.items():
            assert "input" in prices, f"{model} missing 'input' price"
            assert "output" in prices, f"{model} missing 'output' price"


# ---------------------------------------------------------------------------
# ModelRouter dispatch tests (mocked providers)
# ---------------------------------------------------------------------------

class TestModelRouter:

    def _make_router(self) -> ModelRouter:
        return ModelRouter(config_path=MODELS_YAML)

    @pytest.mark.asyncio
    async def test_primary_success(self):
        """Happy path: primary provider responds correctly."""
        router = self._make_router()
        mock_resp = _make_provider_response()

        with patch.object(router, "_dispatch", new=AsyncMock(return_value=mock_resp)):
            resp = await router.complete(
                profile="intake",
                messages=[{"role": "user", "content": "Hello"}],
                system="You are a requirements engineer.",
            )

        assert resp.content == "Test response content"
        assert resp.was_fallback is False
        assert resp.profile == "intake"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """Primary fails → fallback is called → success returned with was_fallback=True."""
        router = self._make_router()
        fallback_resp = _make_provider_response(model="llama-3.3-70b-versatile", provider="groq")

        call_count = 0

        async def mock_dispatch(entry, prof, messages, system, overrides):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("anthropic", "claude-sonnet-4-6", "rate limit")
            return fallback_resp

        with patch.object(router, "_dispatch", new=mock_dispatch):
            resp = await router.complete(
                profile="intake",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert resp.was_fallback is True
        assert resp.provider == "groq"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_both_fail_raises_all_providers_failed(self):
        """Both primary and fallback fail → AllProvidersFailedError raised."""
        router = self._make_router()

        async def always_fail(entry, prof, messages, system, overrides):
            raise ProviderError(entry.provider, entry.model, "server error")

        with patch.object(router, "_dispatch", new=always_fail):
            with pytest.raises(AllProvidersFailedError) as exc_info:
                await router.complete(
                    profile="intake",
                    messages=[{"role": "user", "content": "Hello"}],
                )

        assert "intake" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_profile_raises_key_error(self):
        router = self._make_router()
        with pytest.raises(KeyError, match="not found"):
            await router.complete(
                profile="nonexistent_profile",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_list_profiles(self):
        router = self._make_router()
        profiles = router.list_profiles()
        assert "intake" in profiles
        assert "architecture" in profiles
        assert len(profiles) > 5

    def test_reload_config(self):
        """reload_config() runs without error and keeps profiles intact."""
        router = self._make_router()
        original_profiles = set(router.list_profiles())
        router.reload_config()
        assert set(router.list_profiles()) == original_profiles

    @pytest.mark.asyncio
    async def test_runtime_temperature_override(self):
        """temperature override in complete() is passed through to dispatch."""
        router = self._make_router()
        captured: dict = {}

        async def capture_dispatch(entry, prof, messages, system, overrides):
            captured.update(overrides)
            return _make_provider_response()

        with patch.object(router, "_dispatch", new=capture_dispatch):
            await router.complete(
                profile="intake",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.99,
            )

        assert captured.get("temperature") == 0.99

    @pytest.mark.asyncio
    async def test_cost_calculated_in_response(self):
        """RouterResponse.cost_usd is non-negative and reflects token counts."""
        router = self._make_router()
        mock_resp = _make_provider_response()  # 100 input + 50 output tokens

        with patch.object(router, "_dispatch", new=AsyncMock(return_value=mock_resp)):
            resp = await router.complete(
                profile="intake",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert resp.cost_usd >= 0.0
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
