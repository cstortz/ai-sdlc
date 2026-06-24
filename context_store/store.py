"""
context_store/store.py — ContextStore: unified facade for agents.

Agents import one object and call typed methods — they never touch
asyncpg, neo4j drivers, or redis directly.

Usage:
    from context_store import ContextStore

    async with ContextStore() as store:
        # Audit
        run_id = await store.begin_run(feature_id=fid, agent="intake", layer=1, model="claude-sonnet-4-6", provider="anthropic")
        await store.end_run(run_id, status="completed", cost_usd=0.002)

        # Human gate
        gate_id = await store.request_human_approval(run_id=run_id, feature_id=fid, gate_type="human_approval", message="Review PRD")
        pending = await store.pending_gates(fid)

        # Context handoff
        await store.handoff(feature_id=fid, from_agent="intake", to_agent="architecture", payload={...})
        ctx = await store.receive_handoff(fid, agent="architecture")

        # Semantic search
        results = await store.remember(query="authentication design decisions", limit=5)

        # Graph
        await store.graph.create_prd(id=prd_id, feature_id=fid, file_path="docs/prds/f.md")
        lineage = await store.lineage(fid)
"""
from __future__ import annotations

import logging
from uuid import UUID

from .cache import CacheClient, CHANNEL_GATES, CHANNEL_ALERTS, CHANNEL_INCIDENTS
from .embeddings import EmbeddingClient
from .graph import GraphClient
from .postgres import PostgresClient

logger = logging.getLogger(__name__)


class ContextStore:
    """
    Unified facade over Postgres, Neo4j, Redis, and the embedding client.

    All three backends are connected lazily and share the same lifecycle.
    Use as an async context manager for automatic connect/close, or call
    connect() / close() manually.
    """

    def __init__(
        self,
        *,
        pg_dsn: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        redis_url: str | None = None,
        embed_model: str | None = None,
    ):
        self.pg    = PostgresClient(dsn=pg_dsn)
        self.graph = GraphClient(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        self.cache = CacheClient(url=redis_url)
        self._embedder = EmbeddingClient(model=embed_model) if embed_model else EmbeddingClient()

    async def connect(self) -> None:
        await self.pg.connect()
        await self.graph.connect()
        await self.cache.connect()
        logger.info("ContextStore connected (Postgres + Neo4j + Redis)")

    async def close(self) -> None:
        await self.pg.close()
        await self.graph.close()
        await self.cache.close()
        logger.info("ContextStore closed")

    async def __aenter__(self) -> "ContextStore":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Feature lifecycle
    # ------------------------------------------------------------------

    async def register_feature(
        self,
        *,
        title: str,
        redmine_id: int | None = None,
        description: str | None = None,
    ) -> UUID:
        """
        Create a feature in both Postgres and the graph.
        Returns the new feature UUID.
        """
        fid = await self.pg.upsert_feature(
            title=title, redmine_id=redmine_id, description=description
        )
        await self.graph.create_feature(id=fid, title=title, redmine_id=redmine_id)
        logger.info("Feature registered: %s  title=%r", fid, title)
        return fid

    async def advance_feature(self, feature_id: UUID, status: str) -> None:
        """Update feature status in both Postgres and graph."""
        await self.pg.update_feature_status(feature_id, status)
        await self.graph.update_node_status("Feature", feature_id, status)

    # ------------------------------------------------------------------
    # Agent run audit
    # ------------------------------------------------------------------

    async def begin_run(
        self,
        *,
        feature_id: UUID | None,
        agent: str,
        layer: int,
        model: str,
        provider: str,
        was_fallback: bool = False,
        input_summary: str | None = None,
    ) -> UUID:
        """Log the start of an agent run. Returns run_id."""
        return await self.pg.log_agent_run(
            feature_id=feature_id,
            agent=agent,
            layer=layer,
            model_used=model,
            provider=provider,
            was_fallback=was_fallback,
            input_summary=input_summary,
        )

    async def end_run(
        self,
        run_id: UUID,
        *,
        status: str = "completed",
        output_summary: str | None = None,
        confidence: float | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
    ) -> None:
        await self.pg.complete_agent_run(
            run_id,
            status=status,
            output_summary=output_summary,
            confidence=confidence,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    async def record_decision(
        self,
        *,
        run_id: UUID,
        feature_id: UUID | None,
        agent: str,
        decision_type: str,
        summary: str,
        rationale: str | None = None,
        outcome: str | None = None,
    ) -> UUID:
        return await self.pg.log_decision(
            run_id=run_id,
            feature_id=feature_id,
            agent=agent,
            decision_type=decision_type,
            summary=summary,
            rationale=rationale,
            outcome=outcome,
        )

    # ------------------------------------------------------------------
    # Human gates
    # ------------------------------------------------------------------

    async def request_human_approval(
        self,
        *,
        run_id: UUID,
        feature_id: UUID | None,
        gate_type: str,
        message: str,
        trigger_reason: str = "",
        payload: dict | None = None,
    ) -> UUID:
        """
        Create a human gate in Postgres and publish to the escalation bus.
        Returns the gate UUID.
        """
        gate_id = await self.pg.create_human_gate(
            run_id=run_id,
            feature_id=feature_id,
            gate_type=gate_type,
            trigger_reason=trigger_reason or gate_type,
            message=message,
            payload=payload,
        )
        await self.cache.publish_gate(
            gate_id=gate_id,
            gate_type=gate_type,
            message=message,
            feature_id=feature_id,
        )
        return gate_id

    async def resolve_gate(
        self,
        gate_id: UUID,
        *,
        approved: bool,
        notes: str | None = None,
    ) -> None:
        status = "approved" if approved else "rejected"
        await self.pg.resolve_human_gate(gate_id, status=status, reviewer_notes=notes)

    async def pending_gates(self, feature_id: UUID | None = None) -> list[dict]:
        return await self.pg.get_pending_gates(feature_id)

    # ------------------------------------------------------------------
    # Agent-to-agent context handoff
    # ------------------------------------------------------------------

    async def handoff(
        self,
        *,
        feature_id: UUID,
        from_agent: str,
        to_agent: str,
        payload: dict,
    ) -> UUID:
        """Save context for the next agent in the pipeline."""
        return await self.pg.save_context_snapshot(
            feature_id=feature_id,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
        )

    async def receive_handoff(self, feature_id: UUID, *, agent: str) -> dict | None:
        """Consume and return the context payload left by the previous agent."""
        return await self.pg.consume_context_snapshot(feature_id, agent)

    # ------------------------------------------------------------------
    # Semantic memory (embed + store + search)
    # ------------------------------------------------------------------

    async def memorize(
        self,
        text: str,
        *,
        feature_id: UUID | None = None,
        artifact_type: str,
        artifact_id: str | None = None,
    ) -> int:
        """
        Embed `text` and store all chunks in pgvector.
        Returns the number of chunks stored.
        """
        chunks = await self._embedder.embed_text(
            text,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
        )
        for chunk in chunks:
            await self.pg.store_embedding(
                feature_id=feature_id,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                vector=chunk.vector,
                model_used=self._embedder.model,
            )
        logger.debug("Memorized %d chunk(s): %s/%s", len(chunks), artifact_type, artifact_id)
        return len(chunks)

    async def remember(
        self,
        query: str,
        *,
        limit: int = 5,
        artifact_type: str | None = None,
        feature_id: UUID | None = None,
    ) -> list[dict]:
        """
        Semantic search over stored artifacts.
        Returns top-k matching chunks with similarity scores.
        """
        vector = await self._embedder.embed_single(query)
        return await self.pg.search_embeddings(
            vector,
            limit=limit,
            artifact_type=artifact_type,
            feature_id=feature_id,
        )

    # ------------------------------------------------------------------
    # Graph traversal shortcuts (delegates to self.graph)
    # ------------------------------------------------------------------

    async def lineage(self, feature_id: UUID) -> list[dict]:
        """Full artifact chain: Feature → PRD → ADR → Impl → Tests → Deploy → Incidents."""
        return await self.graph.get_feature_lineage(feature_id)

    async def open_incidents(self) -> list[dict]:
        return await self.graph.get_open_incidents()

    async def recurring_incidents(self, threshold: int = 3) -> list[dict]:
        return await self.graph.get_recurring_incidents(threshold)

    # ------------------------------------------------------------------
    # Distributed locking (delegates to cache)
    # ------------------------------------------------------------------

    def locked(self, resource: str, *, ttl: int = 300):
        """Async context manager — prevents duplicate agent runs."""
        return self.cache.locked(resource, ttl=ttl)
