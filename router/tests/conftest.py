import pytest

# Run all async tests automatically without needing @pytest.mark.asyncio on each one
pytest_plugins = ("pytest_asyncio",)
