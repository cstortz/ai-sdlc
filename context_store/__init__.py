"""
context_store — Unified data layer for the AI SDLC pipeline.

Quick start:
    from context_store import ContextStore

    async with ContextStore() as store:
        fid = await store.register_feature(title="User login")
        run_id = await store.begin_run(feature_id=fid, agent="intake", layer=1,
                                        model="claude-sonnet-4-6", provider="anthropic")
        await store.end_run(run_id, cost_usd=0.001)
"""
from .store import ContextStore
from .postgres import PostgresClient
from .graph import GraphClient
from .cache import CacheClient, CHANNEL_GATES, CHANNEL_ALERTS, CHANNEL_INCIDENTS
from .embeddings import EmbeddingClient

__all__ = [
    "ContextStore",
    "PostgresClient",
    "GraphClient",
    "CacheClient",
    "EmbeddingClient",
    "CHANNEL_GATES",
    "CHANNEL_ALERTS",
    "CHANNEL_INCIDENTS",
]
