"""
context_store/postgres.py — Async Postgres client.

Wraps asyncpg with typed methods for every table in the SDLC schema:
  - features          (work item tracking)
  - agent_runs        (audit trail)
  - decisions         (structured decision log)
  - human_gates       (approval queue)
  - embeddings        (pgvector store — vectors written by embeddings.py)
  - context_snapshots (agent-to-agent handoff payloads)

Connection is acquired from a shared pool; call connect() once at startup
and close() on shutdown, or use as an async context manager.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import asyncpg

logger = logging.getLogger(__name__)

_DSN_ENV = "POSTGRES_DSN"  # e.g. postgresql://sdlc:pass@localhost:5432/sdlc


def _dsn_from_env() -> str:
    dsn = os.environ.get(_DSN_ENV)
    if dsn:
        return dsn
    # Build from individual env vars (matches docker-compose / Helm secrets)
    user = os.environ.get("POSTGRES_USER", "sdlc")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "sdlc")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class PostgresClient:
    """
    Async Postgres client backed by a connection pool.

    Usage:
        pg = PostgresClient()
        await pg.connect()
        run_id = await pg.log_agent_run(feature_id=..., agent="intake", ...)
        await pg.close()

    Or as a context manager:
        async with PostgresClient() as pg:
            ...
    """

    def __init__(self, dsn: str | None = None, min_size: int = 2, max_size: int = 10):
        self._dsn = dsn or _dsn_from_env()
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        logger.info("PostgresClient pool connected (%s–%s conns)", self._min_size, self._max_size)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("PostgresClient pool closed")

    async def __aenter__(self) -> "PostgresClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    @property
    def pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("PostgresClient not connected. Call connect() first.")
        return self._pool

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------

    async def upsert_feature(
        self,
        *,
        title: str,
        redmine_id: int | None = None,
        description: str | None = None,
        status: str = "intake",
        feature_id: UUID | None = None,
    ) -> UUID:
        """Create or update a feature row. Returns the feature UUID."""
        fid = feature_id or uuid4()
        await self.pool.execute(
            """
            INSERT INTO features (id, redmine_id, title, description, status)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id) DO UPDATE
              SET title = EXCLUDED.title,
                  description = EXCLUDED.description,
                  status = EXCLUDED.status,
                  redmine_id = COALESCE(EXCLUDED.redmine_id, features.redmine_id)
            """,
            fid, redmine_id, title, description, status,
        )
        return fid

    async def update_feature_status(self, feature_id: UUID, status: str) -> None:
        await self.pool.execute(
            "UPDATE features SET status = $1 WHERE id = $2",
            status, feature_id,
        )

    async def get_feature(self, feature_id: UUID) -> dict | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM features WHERE id = $1", feature_id
        )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # agent_runs
    # ------------------------------------------------------------------

    async def log_agent_run(
        self,
        *,
        feature_id: UUID | None,
        agent: str,
        layer: int,
        model_used: str,
        provider: str,
        was_fallback: bool = False,
        input_summary: str | None = None,
    ) -> UUID:
        """Insert a new agent_run row and return its UUID."""
        run_id = uuid4()
        await self.pool.execute(
            """
            INSERT INTO agent_runs
              (id, feature_id, agent, layer, model_used, provider,
               was_fallback, status, input_summary)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'running', $8)
            """,
            run_id, feature_id, agent, layer, model_used, provider,
            was_fallback, input_summary,
        )
        logger.debug("Logged agent_run %s  agent=%s  feature=%s", run_id, agent, feature_id)
        return run_id

    async def complete_agent_run(
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
        await self.pool.execute(
            """
            UPDATE agent_runs SET
              status = $1,
              output_summary = $2,
              confidence = $3,
              cost_usd = $4,
              duration_ms = $5,
              error_message = $6,
              completed_at = NOW()
            WHERE id = $7
            """,
            status, output_summary, confidence, cost_usd, duration_ms,
            error_message, run_id,
        )

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------

    async def log_decision(
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
        dec_id = uuid4()
        await self.pool.execute(
            """
            INSERT INTO decisions
              (id, run_id, feature_id, agent, decision_type, summary, rationale, outcome)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            dec_id, run_id, feature_id, agent, decision_type, summary, rationale, outcome,
        )
        return dec_id

    # ------------------------------------------------------------------
    # human_gates
    # ------------------------------------------------------------------

    async def create_human_gate(
        self,
        *,
        run_id: UUID,
        feature_id: UUID | None,
        gate_type: str,
        trigger_reason: str,
        message: str,
        payload: dict | None = None,
    ) -> UUID:
        gate_id = uuid4()
        await self.pool.execute(
            """
            INSERT INTO human_gates
              (id, run_id, feature_id, gate_type, trigger_reason, message, payload)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            gate_id, run_id, feature_id, gate_type, trigger_reason, message,
            json.dumps(payload) if payload else None,
        )
        logger.info("Human gate created: %s  type=%s  feature=%s", gate_id, gate_type, feature_id)
        return gate_id

    async def resolve_human_gate(
        self,
        gate_id: UUID,
        *,
        status: str,   # approved | rejected | expired
        reviewer_notes: str | None = None,
    ) -> None:
        await self.pool.execute(
            """
            UPDATE human_gates
            SET status = $1, reviewer_notes = $2, resolved_at = NOW()
            WHERE id = $3
            """,
            status, reviewer_notes, gate_id,
        )

    async def get_pending_gates(self, feature_id: UUID | None = None) -> list[dict]:
        if feature_id:
            rows = await self.pool.fetch(
                "SELECT * FROM human_gates WHERE status = 'pending' AND feature_id = $1 ORDER BY created_at",
                feature_id,
            )
        else:
            rows = await self.pool.fetch(
                "SELECT * FROM human_gates WHERE status = 'pending' ORDER BY created_at"
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # embeddings (vectors written by embeddings.py, queried here)
    # ------------------------------------------------------------------

    async def store_embedding(
        self,
        *,
        feature_id: UUID | None,
        artifact_type: str,
        artifact_id: str | None,
        chunk_index: int,
        content: str,
        vector: list[float],
        model_used: str = "text-embedding-3-small",
    ) -> UUID:
        emb_id = uuid4()
        # asyncpg needs the vector as a string in pgvector format
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        await self.pool.execute(
            """
            INSERT INTO embeddings
              (id, feature_id, artifact_type, artifact_id, chunk_index,
               content, embedding, model_used)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
            """,
            emb_id, feature_id, artifact_type, artifact_id, chunk_index,
            content, vector_str, model_used,
        )
        return emb_id

    async def search_embeddings(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        artifact_type: str | None = None,
        feature_id: UUID | None = None,
    ) -> list[dict]:
        """Return the top-k most similar chunks by cosine similarity."""
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        where_clauses = []
        params: list[Any] = [vector_str, limit]

        if artifact_type:
            params.append(artifact_type)
            where_clauses.append(f"artifact_type = ${len(params)}")
        if feature_id:
            params.append(feature_id)
            where_clauses.append(f"feature_id = ${len(params)}")

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        rows = await self.pool.fetch(
            f"""
            SELECT id, feature_id, artifact_type, artifact_id, chunk_index,
                   content, model_used, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM embeddings
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            *params,
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # context_snapshots
    # ------------------------------------------------------------------

    async def save_context_snapshot(
        self,
        *,
        feature_id: UUID,
        from_agent: str,
        to_agent: str,
        payload: dict,
    ) -> UUID:
        snap_id = uuid4()
        await self.pool.execute(
            """
            INSERT INTO context_snapshots (id, feature_id, from_agent, to_agent, payload)
            VALUES ($1, $2, $3, $4, $5)
            """,
            snap_id, feature_id, from_agent, to_agent, json.dumps(payload),
        )
        logger.debug("Context snapshot %s: %s → %s", snap_id, from_agent, to_agent)
        return snap_id

    async def consume_context_snapshot(
        self, feature_id: UUID, to_agent: str
    ) -> dict | None:
        """
        Fetch and mark consumed the most recent unconsumed snapshot
        for this feature/agent pair. Returns the payload dict or None.
        """
        row = await self.pool.fetchrow(
            """
            UPDATE context_snapshots
            SET consumed = TRUE, consumed_at = NOW()
            WHERE id = (
                SELECT id FROM context_snapshots
                WHERE feature_id = $1 AND to_agent = $2 AND consumed = FALSE
                ORDER BY created_at DESC
                LIMIT 1
            )
            RETURNING payload
            """,
            feature_id, to_agent,
        )
        if row:
            return json.loads(row["payload"])
        return None
