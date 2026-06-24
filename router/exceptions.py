"""
router/exceptions.py — Custom exceptions for the model router.
"""


class RouterError(Exception):
    """Base class for all router errors."""


class ProviderError(RouterError):
    """Raised when a provider API call fails (after retries)."""
    def __init__(self, provider: str, model: str, message: str):
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}/{model}] {message}")


class AllProvidersFailedError(RouterError):
    """Raised when both primary and fallback providers fail."""
    def __init__(self, profile: str, primary_error: Exception, fallback_error: Exception):
        self.profile = profile
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        super().__init__(
            f"All providers failed for profile '{profile}'.\n"
            f"  Primary:  {primary_error}\n"
            f"  Fallback: {fallback_error}"
        )


class CostThresholdExceededError(RouterError):
    """Raised when estimated cost exceeds the profile's cost_threshold_usd."""
    def __init__(self, profile: str, estimated: float, threshold: float):
        self.profile = profile
        self.estimated = estimated
        self.threshold = threshold
        super().__init__(
            f"Estimated cost ${estimated:.4f} exceeds threshold ${threshold:.4f} "
            f"for profile '{profile}'. Falling back to cheaper model."
        )


class ProfileNotFoundError(RouterError):
    """Raised when a requested profile does not exist in models.yaml."""
