"""
context_store/tests/test_context_store.py

Unit tests using mocked backends — no live Postgres, Neo4j, or Redis needed.
Run: pytest context_store/tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import UUID, uuid4

import pytest

from context_store.postgres import PostgresClient
from context_store.graph import GraphClient, _now
from context_store.cache import CacheClient, CHANNEL_GATES
from context_store.embeddings import EmbeddingClient, _split_text
from context_store.store import ContextStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> UUID:
    return uuid4()


# ---------------------------------------------------------------------------
# Text chunking (pure function — no mocks needed)
# ---------------------------------------------------------------------------

class TestSplitText:

    def test_short_text_single_chunk(self):
        chunks = _split_text("hello world", chunk_size=1000, overlap=100)
        assert chunks == ["hello world"]

    def test_empty_text(self):
        assert _split_text("", chunk_size=1000, overlap=100) == []

    def test_long_text_splits(self):
        words = ["word"] * 2500
        text = " ".join(words)
        chunks = _split_text(text, chunk_size=1000, overlap=100)
        assert len(chunks) > 1
        # All chunks within size limit
        for c in chunks:
            assert len(c.split()) <= 1000

    def test_overlap_present(self):
        words = [str(i) for i in range(1200)]
        text = " ".join(words)
        chunks = _split_text(text, chunk_size=1000, overlap=100)
        # First chunk ends at word 999, second starts at word 900
        first_end = chunks[0].split()[-1]
        second_start = chunks[1].split()[0]
        # There should be overlap
        assert first_end in chunks[1]


# ---------------------------------------------------------------------------
# PostgresClient (mocked pool)
# ---------------------------------------------------------------------------

class TestPostgresClient:

    def _make_pg(self) -> PostgresClient:
        pg = PostgresClient(dsn="postgresql://test/test")
        pg._pool = AsyncMock()
        return pg

    @pytest.mark.asyncio
    async def test_upsert_feature_returns_uuid(self):
        pg = self._make_pg()
        pg._pool.execute = AsyncMock(return_value=None)
        fid = await pg.upsert_feature(title="Test feature")
        assert isinstance(fid, UUID)
        pg._pool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_feature_uses_provided_id(self):
        pg = self._make_pg()
        pg._pool.execute = AsyncMock(return_value=None)
        provided_id = _uuid()
        result = await pg.upsert_feature(title="Test", feature_id=provided_id)
        assert result == provided_id

    @pytest.mark.asyncio
    async def test_log_agent_run_returns_uuid(self):
        pg = self._make_pg()
        pg._pool.execute = AsyncMock(return_value=None)
        run_id = await pg.log_agent_run(
            feature_id=_uuid(), agent="intake", layer=1,
            model_used="claude-sonnet-4-6", provider="anthropic",
        )
        assert isinstance(run_id, UUID)

    @pytest.mark.asyncio
    async def test_complete_agent_run(self):
        pg = self._make_pg()
        pg._pool.execute = AsyncMock(return_value=None)
        run_id = _uuid()
        await pg.complete_agent_run(run_id, status="completed", cost_usd=0.002, duration_ms=1500)
        pg._pool.execute.assert_called_once()
        call_args = pg._pool.execute.call_args[0]
        assert "completed" in call_args

    @pytest.mark.asyncio
    async def test_create_human_gate_returns_uuid(self):
        pg = self._make_pg()
        pg._pool.execute = AsyncMock(return_value=None)
        gate_id = await pg.create_human_gate(
            run_id=_uuid(), feature_id=_uuid(),
            gate_type="human_approval",
            trigger_reason="always",
            message="Review PRD",
        )
        assert isinstance(gate_id, UUID)

    @pytest.mark.asyncio
    async def test_get_pending_gates_with_feature(self):
        pg = self._make_pg()
        mock_row = {"id": str(_uuid()), "status": "pending", "message": "Review"}
        pg._pool.fetch = AsyncMock(return_value=[mock_row])
        result = await pg.get_pending_gates(_uuid())
        assert len(result) == 1
        assert result[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_consume_context_snapshot_returns_payload(self):
        pg = self._make_pg()
        payload = {"prd": "content", "feature_id": str(_uuid())}
        mock_row = {"payload": json.dumps(payload)}
        pg._pool.fetchrow = AsyncMock(return_value=mock_row)
        result = await pg.consume_context_snapshot(_uuid(), "architecture")
        assert result == payload

    @pytest.mark.asyncio
    async def test_consume_context_snapshot_returns_none_when_empty(self):
        pg = self._make_pg()
        pg._pool.fetchrow = AsyncMock(return_value=None)
        result = await pg.consume_context_snapshot(_uuid(), "architecture")
        assert result is None


# ---------------------------------------------------------------------------
# GraphClient (mocked driver)
# ---------------------------------------------------------------------------

class TestGraphClient:

    def _make_graph(self) -> GraphClient:
        graph = GraphClient()
        graph._driver = AsyncMock()
        # Mock session context manager
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.__aiter__ = AsyncMock(return_value=iter([]))
        mock_session.run = AsyncMock(return_value=mock_result)
        graph._driver.session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=None),
        ))
        return graph

    @pytest.mark.asyncio
    async def test_create_feature_calls_driver(self):
        graph = self._make_graph()
        fid = _uuid()
        await graph.create_feature(id=fid, title="Login feature")
        assert graph._driver.session.called

    @pytest.mark.asyncio
    async def test_update_node_status(self):
        graph = self._make_graph()
        await graph.update_node_status("Feature", _uuid(), "architecture")
        assert graph._driver.session.called

    def test_now_is_iso_string(self):
        ts = _now()
        assert "T" in ts and "+" in ts or "Z" in ts or ts.endswith("+00:00")


# ---------------------------------------------------------------------------
# CacheClient (mocked redis)
# ---------------------------------------------------------------------------

class TestCacheClient:

    def _make_cache(self) -> CacheClient:
        cache = CacheClient(url="redis://localhost/0")
        cache._client = AsyncMock()
        return cache

    @pytest.mark.asyncio
    async def test_set_and_get_agent_state(self):
        cache = self._make_cache()
        state = {"step": "interview", "questions_asked": 2}
        cache._client.set = AsyncMock(return_value=True)
        cache._client.get = AsyncMock(return_value=json.dumps(state))

        await cache.set_agent_state("intake:abc", state, ttl=600)
        result = await cache.get_agent_state("intake:abc")

        assert result == state
        cache._client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_missing_state_returns_none(self):
        cache = self._make_cache()
        cache._client.get = AsyncMock(return_value=None)
        result = await cache.get_agent_state("intake:missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_acquire_lock_returns_true_when_available(self):
        cache = self._make_cache()
        cache._client.set = AsyncMock(return_value=True)
        acquired = await cache.acquire_lock("feature:abc", ttl=60)
        assert acquired is True

    @pytest.mark.asyncio
    async def test_acquire_lock_returns_false_when_held(self):
        cache = self._make_cache()
        cache._client.set = AsyncMock(return_value=None)  # nx=True returns None when key exists
        acquired = await cache.acquire_lock("feature:abc", ttl=60)
        assert acquired is False

    @pytest.mark.asyncio
    async def test_publish_gate_calls_publish(self):
        cache = self._make_cache()
        cache._client.publish = AsyncMock(return_value=1)
        gate_id = _uuid()
        fid = _uuid()
        await cache.publish_gate(gate_id=gate_id, gate_type="human_approval", message="Review", feature_id=fid)
        cache._client.publish.assert_called_once()
        call_args = cache._client.publish.call_args[0]
        assert call_args[0] == CHANNEL_GATES
        payload = json.loads(call_args[1])
        assert payload["gate_type"] == "human_approval"
        assert payload["gate_id"] == str(gate_id)


# ---------------------------------------------------------------------------
# ContextStore facade (mocked all backends)
# ---------------------------------------------------------------------------

class TestContextStore:

    def _make_store(self) -> ContextStore:
        store = ContextStore()
        store.pg    = AsyncMock(spec=PostgresClient)
        store.graph = AsyncMock(spec=GraphClient)
        store.cache = AsyncMock(spec=CacheClient)
        store._embedder = AsyncMock(spec=EmbeddingClient)
        return store

    @pytest.mark.asyncio
    async def test_register_feature_calls_both_backends(self):
        store = self._make_store()
        fid = _uuid()
        store.pg.upsert_feature = AsyncMock(return_value=fid)
        store.graph.create_feature = AsyncMock(return_value=None)

        result = await store.register_feature(title="Login feature")

        assert result == fid
        store.pg.upsert_feature.assert_called_once()
        store.graph.create_feature.assert_called_once()

    @pytest.mark.asyncio
    async def test_begin_run_delegates_to_pg(self):
        store = self._make_store()
        run_id = _uuid()
        store.pg.log_agent_run = AsyncMock(return_value=run_id)

        result = await store.begin_run(
            feature_id=_uuid(), agent="intake", layer=1,
            model="claude-sonnet-4-6", provider="anthropic",
        )
        assert result == run_id

    @pytest.mark.asyncio
    async def test_request_human_approval_publishes_to_bus(self):
        store = self._make_store()
        gate_id = _uuid()
        store.pg.create_human_gate = AsyncMock(return_value=gate_id)
        store.cache.publish_gate = AsyncMock(return_value=None)

        result = await store.request_human_approval(
            run_id=_uuid(), feature_id=_uuid(),
            gate_type="human_approval", message="Review PRD",
        )
        assert result == gate_id
        store.cache.publish_gate.assert_called_once()

    @pytest.mark.asyncio
    async def test_memorize_embeds_and_stores(self):
        store = self._make_store()
        mock_chunk = MagicMock(chunk_index=0, content="text", vector=[0.1] * 1536)
        store._embedder.embed_text = AsyncMock(return_value=[mock_chunk])
        store.pg.store_embedding = AsyncMock(return_value=_uuid())
        store._embedder.model = "text-embedding-3-small"

        count = await store.memorize("Some PRD text", artifact_type="prd")

        assert count == 1
        store._embedder.embed_text.assert_called_once()
        store.pg.store_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_embeds_query_then_searches(self):
        store = self._make_store()
        store._embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)
        store.pg.search_embeddings = AsyncMock(return_value=[{"content": "match", "similarity": 0.95}])

        results = await store.remember("authentication flow")

        assert len(results) == 1
        store._embedder.embed_single.assert_called_once_with("authentication flow")

    @pytest.mark.asyncio
    async def test_handoff_and_receive(self):
        store = self._make_store()
        snap_id = _uuid()
        fid = _uuid()
        payload = {"prd": "content"}
        store.pg.save_context_snapshot = AsyncMock(return_value=snap_id)
        store.pg.consume_context_snapshot = AsyncMock(return_value=payload)

        saved = await store.handoff(feature_id=fid, from_agent="intake", to_agent="architecture", payload=payload)
        received = await store.receive_handoff(fid, agent="architecture")

        assert saved == snap_id
        assert received == payload
