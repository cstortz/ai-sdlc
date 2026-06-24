"""
agents/architecture/tests/test_architecture.py

Unit tests for the Architecture Agent and ADR writer.
No live API calls, no live DB — everything is mocked.

Run: pytest agents/architecture/tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agents.base import AgentStatus
from agents.architecture.agent import ArchitectureAgent, _gate_trigger, _gate_message, _read_prd
from agents.architecture.adr_writer import (
    ADRWriter,
    ADRResult,
    _clean_content,
    _detect_breaking_change,
    _detect_complexity,
)
from router import RouterResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_ADR = """\
# ADR: User Authentication

**Status:** Proposed
**Date:** 2026-06-23
**Feature:** 12345678-1234-1234-1234-123456789012

## Context
No authentication system exists. Users need secure access.

## Decision
Implement JWT-based authentication using PyJWT.

## Consequences
### Positive
- Stateless tokens reduce server load.

### Negative / Trade-offs
- Token revocation requires a blocklist.

### Risks
- Token leakage if transport is not encrypted.

## Alternatives Considered
Session-based auth was rejected due to horizontal scaling complexity.

## Implementation Notes
- Data models: add `users` table with hashed passwords
- Security considerations: use bcrypt for hashing, TLS for transport
- Estimated complexity: Medium

## Breaking Change Assessment
NO — this is a new feature with no existing API contract changes.
"""

_SAMPLE_ADR_BREAKING = """\
# ADR: Replace REST with GraphQL

## Implementation Notes
- Estimated complexity: High

## Breaking Change Assessment
YES — existing REST endpoints will be removed, breaking all current API clients.
"""


def _router_response(content: str = _SAMPLE_ADR) -> RouterResponse:
    return RouterResponse(
        content=content,
        profile="architecture",
        model_used="claude-sonnet-4-6",
        provider="anthropic",
        was_fallback=False,
        input_tokens=200,
        output_tokens=400,
        cost_usd=0.002,
        duration_ms=800,
    )


def _mock_store(feature_id: UUID, handoff_payload: dict | None = None) -> MagicMock:
    store = MagicMock()
    store.receive_handoff = AsyncMock(return_value=handoff_payload or {
        "prd_file": "/tmp/fake_prd.md",
        "prd_id": str(uuid4()),
        "title": "User Authentication",
        "redmine_id": None,
    })
    store.advance_feature = AsyncMock(return_value=None)
    store.graph = MagicMock()
    store.graph.create_adr = AsyncMock(return_value=None)
    store.memorize = AsyncMock(return_value=2)
    store.begin_run = AsyncMock(return_value=uuid4())
    store.end_run = AsyncMock(return_value=None)
    store.request_human_approval = AsyncMock(return_value=uuid4())
    store.record_decision = AsyncMock(return_value=uuid4())
    store.handoff = AsyncMock(return_value=uuid4())
    store.remember = AsyncMock(return_value=[])
    store.cache = MagicMock()
    store.cache.client = MagicMock()
    store.cache.client.publish = AsyncMock(return_value=1)

    @asynccontextmanager
    async def _locked(*args, **kwargs):
        yield True

    store.locked = _locked
    return store


def _mock_router(content: str = _SAMPLE_ADR) -> MagicMock:
    router = MagicMock()
    router.complete = AsyncMock(return_value=_router_response(content))
    return router


def _make_agent(
    feature_id: UUID | None = None,
    adr_content: str = _SAMPLE_ADR,
    handoff_payload: dict | None = None,
) -> tuple[ArchitectureAgent, MagicMock, UUID]:
    fid = feature_id or uuid4()
    store = _mock_store(fid, handoff_payload)
    router = _mock_router(adr_content)
    agent = ArchitectureAgent(store=store, router=router)
    return agent, store, fid


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestDetectBreakingChange:

    def test_yes_is_breaking(self):
        assert _detect_breaking_change(_SAMPLE_ADR_BREAKING) is True

    def test_no_is_not_breaking(self):
        assert _detect_breaking_change(_SAMPLE_ADR) is False

    def test_missing_section_returns_false(self):
        assert _detect_breaking_change("# ADR: Foo\n\nSome content.") is False

    def test_case_insensitive_yes(self):
        content = "## Breaking Change Assessment\nYES — breaking."
        assert _detect_breaking_change(content) is True

    def test_narrative_yes_not_matched(self):
        # Only YES at line start of section counts
        content = "## Breaking Change Assessment\nNO, but consider that YES scenarios exist."
        assert _detect_breaking_change(content) is False


class TestDetectComplexity:

    def test_medium_detected(self):
        assert _detect_complexity(_SAMPLE_ADR) == "Medium"

    def test_high_detected(self):
        assert _detect_complexity(_SAMPLE_ADR_BREAKING) == "High"

    def test_low_detected(self):
        content = "- Estimated complexity: Low"
        assert _detect_complexity(content) == "Low"

    def test_missing_returns_unknown(self):
        assert _detect_complexity("# ADR: Foo") == "Unknown"

    def test_bold_markers_stripped(self):
        content = "- Estimated complexity: **High**"
        assert _detect_complexity(content) == "High"


class TestCleanContent:

    def test_strips_markdown_fence(self):
        raw = "```markdown\n# ADR: Test\n```"
        assert _clean_content(raw) == "# ADR: Test"

    def test_strips_plain_fence(self):
        raw = "```\n# ADR: Test\n```"
        assert _clean_content(raw) == "# ADR: Test"

    def test_passthrough_clean_content(self):
        assert _clean_content("# ADR: Test") == "# ADR: Test"


class TestGateTrigger:

    def _adr(self, is_breaking: bool = False, complexity: str = "Medium") -> ADRResult:
        return ADRResult(
            feature_id=uuid4(),
            title="T",
            content="",
            file_path=Path("/tmp/test.md"),
            is_breaking_change=is_breaking,
            complexity=complexity,
            router_response=_router_response(),
        )

    def test_always_escalate_triggers(self):
        assert _gate_trigger(self._adr(), always_escalate=True) is not None

    def test_breaking_change_triggers(self):
        assert _gate_trigger(self._adr(is_breaking=True), always_escalate=False) is not None

    def test_high_complexity_triggers(self):
        assert _gate_trigger(self._adr(complexity="High"), always_escalate=False) is not None

    def test_low_risk_no_gate(self):
        assert _gate_trigger(self._adr(is_breaking=False, complexity="Medium"), always_escalate=False) is None

    def test_gate_message_includes_reason(self):
        adr = self._adr(is_breaking=True)
        msg = _gate_message(adr, "Auth", always_escalate=False)
        assert "breaking change" in msg.lower()
        assert "Auth" in msg


class TestReadPrd:

    def test_reads_existing_file(self, tmp_path):
        prd = tmp_path / "prd.md"
        prd.write_text("# PRD: Test")
        assert _read_prd(str(prd)) == "# PRD: Test"

    def test_missing_file_returns_none(self):
        assert _read_prd("/nonexistent/path.md") is None


# ---------------------------------------------------------------------------
# ADRWriter
# ---------------------------------------------------------------------------


class TestADRWriter:

    def _writer(self, content: str = _SAMPLE_ADR) -> tuple[ADRWriter, MagicMock]:
        router = _mock_router(content)
        return ADRWriter(router), router

    @pytest.mark.asyncio
    async def test_write_calls_router_and_returns_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        writer, router = self._writer()

        result = await writer.write(
            feature_id=uuid4(),
            title="User Auth",
            prd_content="# PRD: User Auth\n\nProblem statement.",
        )

        router.complete.assert_called_once()
        assert result.file_path.exists()
        assert "ADR" in result.content
        assert result.is_breaking_change is False
        assert result.complexity == "Medium"

    @pytest.mark.asyncio
    async def test_write_detects_breaking_change(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        writer, _ = self._writer(_SAMPLE_ADR_BREAKING)

        result = await writer.write(
            feature_id=uuid4(),
            title="GraphQL Migration",
            prd_content="# PRD: GraphQL Migration",
        )

        assert result.is_breaking_change is True
        assert result.complexity == "High"

    @pytest.mark.asyncio
    async def test_write_uses_architecture_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        writer, router = self._writer()
        await writer.write(feature_id=uuid4(), title="T", prd_content="PRD")
        call_args = router.complete.call_args
        assert call_args[1].get("profile") == "architecture" or call_args[0][0] == "architecture"

    @pytest.mark.asyncio
    async def test_revise_writes_versioned_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        writer, _ = self._writer("# ADR: Revised\n\n## Breaking Change Assessment\nNO.\n\n## Implementation Notes\n- Estimated complexity: Low\n")

        result = await writer.revise(
            feature_id=uuid4(),
            title="Auth",
            existing_adr=_SAMPLE_ADR,
            feedback="Clarify the risk section",
            version=2,
        )

        assert "_v2" in result.file_path.name
        assert result.file_path.exists()

    @pytest.mark.asyncio
    async def test_write_includes_past_context_in_messages(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        writer, router = self._writer()

        await writer.write(
            feature_id=uuid4(),
            title="T",
            prd_content="PRD",
            existing_context="[Past ADR] JWT auth was used in project X.",
        )

        messages = router.complete.call_args[1]["messages"]
        assert any("Past ADR" in m["content"] for m in messages)


# ---------------------------------------------------------------------------
# ArchitectureAgent integration
# ---------------------------------------------------------------------------


class TestArchitectureAgent:

    @pytest.mark.asyncio
    async def test_run_creates_gate_in_phase1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        monkeypatch.setattr("agents.architecture.agent._read_prd", lambda _: "# PRD: Auth\n\nTest.")

        agent, store, fid = _make_agent()
        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.GATE_WAIT
        assert result.gate_id is not None
        store.request_human_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_fails_without_handoff(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        agent, store, fid = _make_agent()
        store.receive_handoff = AsyncMock(return_value=None)

        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.FAILED
        assert "No handoff" in result.error

    @pytest.mark.asyncio
    async def test_run_fails_if_prd_unreadable(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        monkeypatch.setattr("agents.architecture.agent._read_prd", lambda _: None)

        agent, store, fid = _make_agent()
        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.FAILED
        assert "PRD" in result.error

    @pytest.mark.asyncio
    async def test_run_creates_adr_graph_node(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        monkeypatch.setattr("agents.architecture.agent._read_prd", lambda _: "# PRD: Auth\n\nTest.")

        agent, store, fid = _make_agent()
        await agent.run(feature_id=fid)

        store.graph.create_adr.assert_called_once()
        call_kwargs = store.graph.create_adr.call_args[1]
        assert call_kwargs["feature_id"] == fid

    @pytest.mark.asyncio
    async def test_run_stores_embeddings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        monkeypatch.setattr("agents.architecture.agent._read_prd", lambda _: "# PRD: Auth\n\nTest.")

        agent, store, fid = _make_agent()
        await agent.run(feature_id=fid)

        store.memorize.assert_called_once()
        call_args = store.memorize.call_args[1]
        assert call_args["artifact_type"] == "adr"

    @pytest.mark.asyncio
    async def test_run_creates_handoff_to_implementation(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        monkeypatch.setattr("agents.architecture.agent._read_prd", lambda _: "# PRD: Auth\n\nTest.")

        agent, store, fid = _make_agent()
        await agent.run(feature_id=fid)

        store.handoff.assert_called_once()
        kwargs = store.handoff.call_args[1]
        assert kwargs["from_agent"] == "architecture"
        assert kwargs["to_agent"] == "implementation"

    @pytest.mark.asyncio
    async def test_run_skipped_when_lock_held(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        monkeypatch.setattr("agents.architecture.agent._read_prd", lambda _: "# PRD: Auth\n\nTest.")

        agent, store, fid = _make_agent()

        @asynccontextmanager
        async def _locked_false(*args, **kwargs):
            yield False

        store.locked = _locked_false
        result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_auto_approve_non_breaking_in_phase2(self, tmp_path, monkeypatch):
        """
        Simulate Phase 2 behaviour: always_escalate=False, non-breaking, Medium complexity.
        Agent should succeed without creating a gate.
        """
        monkeypatch.setattr("agents.architecture.adr_writer.ADR_DIR", tmp_path)
        monkeypatch.setattr("agents.architecture.agent._read_prd", lambda _: "# PRD: Auth\n\nTest.")

        # Patch workflow config to disable always_escalate
        agent, store, fid = _make_agent(adr_content=_SAMPLE_ADR)
        agent._workflow = {
            "stages": {
                "architecture": {"breaking_change_always_escalate": False}
            }
        }

        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.SUCCESS
        store.request_human_approval.assert_not_called()
