"""
router — Model routing layer for the AI SDLC pipeline.

Quick start:
    from router import ModelRouter

    router = ModelRouter()
    response = await router.complete(
        profile="intake",
        messages=[{"role": "user", "content": "Build a login feature"}],
        system="You are a requirements engineer.",
    )
"""
from .router import ModelRouter, RouterResponse
from .config import RouterConfig, PRICING, estimate_cost
from .exceptions import (
    RouterError,
    ProviderError,
    AllProvidersFailedError,
    CostThresholdExceededError,
)

__all__ = [
    "ModelRouter",
    "RouterResponse",
    "RouterConfig",
    "PRICING",
    "estimate_cost",
    "RouterError",
    "ProviderError",
    "AllProvidersFailedError",
    "CostThresholdExceededError",
]
