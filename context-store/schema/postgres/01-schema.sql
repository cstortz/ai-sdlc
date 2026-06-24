-- =============================================================================
-- AI SDLC — Postgres Schema
-- Database: sdlc
-- =============================================================================
-- Tables:
--   features          — top-level work items (synced from Redmine)
--   agent_runs        — audit log of every agent invocation
--   decisions         — structured record of every agent decision
--   human_gates       — pending and resolved human approval requests
--   embeddings        — pgvector store for semantic search over artifacts
--   context_snapshots — point-in-time context passed between agents
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------------
-- features
-- Top-level work items. Sourced from Redmine; agents reference feature_id.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    redmine_id    INTEGER UNIQUE,                  -- Redmine issue ID (null if NL-prompt origin)
    title         TEXT NOT NULL,
    description   TEXT,
    status        TEXT NOT NULL DEFAULT 'intake',  -- intake | architecture | implementation | testing | review | deploy | done
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- agent_runs
-- Every time an agent runs, one row is inserted here.
-- This is the primary audit trail.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_runs (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_id     UUID REFERENCES features(id) ON DELETE SET NULL,
    agent          TEXT NOT NULL,                  -- intake | architecture | implementation | ...
    layer          SMALLINT NOT NULL,              -- 1–7
    model_used     TEXT NOT NULL,                  -- actual model string used
    provider       TEXT NOT NULL,                  -- anthropic | groq | together | ...
    was_fallback   BOOLEAN NOT NULL DEFAULT FALSE, -- TRUE if primary model was unavailable
    status         TEXT NOT NULL DEFAULT 'running', -- running | completed | failed | escalated
    input_summary  TEXT,
    output_summary TEXT,
    confidence     NUMERIC(4,3),                   -- 0.000–1.000
    cost_usd       NUMERIC(10,6),
    duration_ms    INTEGER,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at   TIMESTAMPTZ,
    error_message  TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_feature ON agent_runs(feature_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_agent   ON agent_runs(agent);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status  ON agent_runs(status);

-- ---------------------------------------------------------------------------
-- decisions
-- Structured record of significant agent decisions (architecture choices,
-- model selection overrides, gate evaluations, etc.)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id        UUID REFERENCES agent_runs(id) ON DELETE CASCADE,
    feature_id    UUID REFERENCES features(id) ON DELETE SET NULL,
    agent         TEXT NOT NULL,
    decision_type TEXT NOT NULL,  -- architecture | test_strategy | security_finding | model_override | ...
    summary       TEXT NOT NULL,
    rationale     TEXT,
    outcome       TEXT,           -- accepted | rejected | deferred | overridden_by_human
    decided_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decisions_feature ON decisions(feature_id);
CREATE INDEX IF NOT EXISTS idx_decisions_type    ON decisions(decision_type);

-- ---------------------------------------------------------------------------
-- human_gates
-- Pending and resolved human approval requests.
-- Agents INSERT with status='pending'; humans resolve to 'approved'/'rejected'.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS human_gates (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id          UUID REFERENCES agent_runs(id) ON DELETE CASCADE,
    feature_id      UUID REFERENCES features(id) ON DELETE SET NULL,
    gate_type       TEXT NOT NULL,  -- human_approval | cost_review | security_review | architecture_review
    trigger_reason  TEXT NOT NULL,
    message         TEXT NOT NULL,
    payload         JSONB,          -- relevant context for the human reviewer
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | expired
    reviewer_notes  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_human_gates_feature ON human_gates(feature_id);
CREATE INDEX IF NOT EXISTS idx_human_gates_status  ON human_gates(status);

-- ---------------------------------------------------------------------------
-- embeddings
-- pgvector store. One row per artifact chunk.
-- Used for semantic retrieval by agents ("what past decisions are relevant here?")
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_id      UUID REFERENCES features(id) ON DELETE SET NULL,
    artifact_type   TEXT NOT NULL,  -- prd | adr | spec | code_chunk | test | incident_report
    artifact_id     TEXT,           -- file path or graph node ID
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    content         TEXT NOT NULL,
    embedding       vector(1536),   -- dimension matches text-embedding-3-small; adjust for other models
    model_used      TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_feature       ON embeddings(feature_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_artifact_type ON embeddings(artifact_type);
-- Vector similarity index (IVFFlat — switch to HNSW for large datasets)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- context_snapshots
-- Serialized context bundle passed from one agent to the next.
-- Provides a recoverable handoff record if an agent crashes mid-pipeline.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS context_snapshots (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_id      UUID REFERENCES features(id) ON DELETE CASCADE,
    from_agent      TEXT NOT NULL,
    to_agent        TEXT NOT NULL,
    payload         JSONB NOT NULL,  -- full context blob
    consumed        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_context_snapshots_feature   ON context_snapshots(feature_id);
CREATE INDEX IF NOT EXISTS idx_context_snapshots_consumed  ON context_snapshots(consumed);

-- ---------------------------------------------------------------------------
-- Trigger: auto-update features.updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_features_updated_at
    BEFORE UPDATE ON features
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
