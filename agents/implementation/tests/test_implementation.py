"""
agents/implementation/tests/test_implementation.py

Unit tests for the Implementation Agent and CodeWriter.
No live API calls, no live DB — everything is mocked.

Run: pytest agents/implementation/tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agents.base import AgentStatus
from agents.implementation.agent import (
    ImplementationAgent,
    _detect_language,
    _read_file,
)
from agents.implementation.code_writer import (
    CodeWriter,
    GeneratedFile,
    ImplementationResult,
    _build_messages,
    _parse_response,
)
from router import RouterResponse


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_VALID_JSON = {
    "summary": "Implemented JWT authentication module.",
    "files": [
        {
            "path": "src/auth/jwt.py",
            "description": "JWT encode/decode helpers.",
            "content": "# jwt.py\nimport jwt\n\ndef encode(payload): ...\ndef decode(token): ...\n",
        },
        {
            "path": "src/auth/__init__.py",
            "description": "Package init.",
            "content": "# init\n",
        },
    ],
    "notes": "Uses PyJWT library.",
    "estimated_test_coverage": "Test encode, decode, and expiry handling.",
}

_ADR_CONTENT = """\
# ADR: User Authentication
## Decision
Use JWT via PyJWT.
## Breaking Change Assessment
NO.
"""

_PRD_CONTENT = "# PRD: User Authentication\n\n## Problem Statement\nNo auth exists."


def _router_response(content: str | None = None) -> RouterResponse:
    if content is None:
        content = json.dumps(_VALID_JSON)
    return RouterResponse(
        content=content,
        profile="implementation",
        model_used="claude-sonnet-4-6",
        provider="anthropic",
        was_fallback=False,
        input_tokens=300,
        output_tokens=600,
        cost_usd=0.004,
        duration_ms=1200,
    )


def _mock_router(content: str | None = None) -> MagicMock:
    router = MagicMock()
    router.complete = AsyncMock(return_value=_router_response(content))
    return router


def _mock_store(feature_id: UUID, handoff_payload: dict | None = None) -> MagicMock:
    store = MagicMock()
    store.receive_handoff = AsyncMock(return_value=handoff_payload or {
        "adr_file": "/tmp/fake_adr.md",
        "adr_id": str(uuid4()),
        "title": "User Authentication",
        "is_breaking_change": False,
        "complexity": "Medium",
        "redmine_id": None,
    })
    store.advance_feature = AsyncMock(return_value=None)
    store.graph = MagicMock()
    store.graph.create_implementation = AsyncMock(return_value=None)
    store.memorize = AsyncMock(return_value=3)
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


def _make_agent(
    feature_id: UUID | None = None,
    handoff_payload: dict | None = None,
    router_content: str | None = None,
) -> tuple[ImplementationAgent, MagicMock, UUID]:
    fid = feature_id or uuid4()
    store = _mock_store(fid, handoff_payload)
    router = _mock_router(router_content)
    agent = ImplementationAgent(store=store, router=router)
    return agent, store, fid


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestDetectLanguage:

    def test_python(self):
        assert _detect_language(["src/auth/jwt.py", "src/auth/__init__.py"]) == "python"

    def test_typescript(self):
        assert _detect_language(["src/app.ts", "src/types.ts"]) == "typescript"

    def test_mixed_picks_majority(self):
        result = _detect_language(["src/a.py", "src/b.py", "src/c.ts"])
        assert result == "python"

    def test_empty_returns_unknown(self):
        assert _detect_language([]) == "unknown"

    def test_sql_migration(self):
        assert _detect_language(["src/migrations/001.sql"]) == "sql"


class TestReadFile:

    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Hello")
        assert _read_file(str(f)) == "# Hello"

    def test_missing_returns_none(self):
        assert _read_file("/no/such/file.md") is None


class TestBuildMessages:

    def test_includes_prd_and_adr(self):
        messages = _build_messages("Auth", "PRD content", "ADR content", None)
        assert len(messages) == 1
        body = messages[0]["content"]
        assert "PRD content" in body
        assert "ADR content" in body

    def test_includes_past_context_when_provided(self):
        messages = _build_messages("Auth", "PRD", "ADR", "Past impl context")
        body = messages[0]["content"]
        assert "Past impl context" in body


class TestParseResponse:

    def test_parses_valid_json(self):
        response = _router_response(json.dumps(_VALID_JSON))
        result = _parse_response(response, uuid4(), "Auth", 1)
        assert result.summary == "Implemented JWT authentication module."
        assert len(result.files) == 2
        assert result.files[0].path == "src/auth/jwt.py"

    def test_parses_json_inside_code_fence(self):
        fenced = f"```json\n{json.dumps(_VALID_JSON)}\n```"
        response = _router_response(fenced)
        result = _parse_response(response, uuid4(), "Auth", 1)
        assert len(result.files) == 2

    def test_raises_on_invalid_json(self):
        response = _router_response("not json at all")
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_response(response, uuid4(), "Auth", 1)

    def test_raises_on_missing_files_key(self):
        response = _router_response(json.dumps({"summary": "x"}))
        with pytest.raises(ValueError, match="missing required 'files'"):
            _parse_response(response, uuid4(), "Auth", 1)

    def test_sets_iteration(self):
        response = _router_response(json.dumps(_VALID_JSON))
        result = _parse_response(response, uuid4(), "Auth", 3)
        assert result.iteration == 3


# ---------------------------------------------------------------------------
# CodeWriter
# ---------------------------------------------------------------------------


class TestCodeWriter:

    @pytest.mark.asyncio
    async def test_write_calls_router_and_creates_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        router = _mock_router()
        writer = CodeWriter(router)

        result = await writer.write(
            feature_id=uuid4(),
            title="Auth",
            prd_content=_PRD_CONTENT,
            adr_content=_ADR_CONTENT,
        )

        router.complete.assert_called_once()
        assert len(result.files) == 2
        assert result.summary == "Implemented JWT authentication module."
        # Files should be written to disk
        for f in result.files:
            assert f.abs_path.exists()

    @pytest.mark.asyncio
    async def test_write_uses_implementation_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        router = _mock_router()
        writer = CodeWriter(router)

        await writer.write(feature_id=uuid4(), title="T", prd_content="P", adr_content="A")

        call_kwargs = router.complete.call_args[1]
        assert call_kwargs.get("profile") == "implementation"

    @pytest.mark.asyncio
    async def test_revise_increments_iteration(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        router = _mock_router()
        writer = CodeWriter(router)

        prev_result = ImplementationResult(
            feature_id=uuid4(),
            title="Auth",
            summary="Initial",
            files=[GeneratedFile(path="src/auth/jwt.py", description="JWT", content="# code")],
            notes="",
            estimated_test_coverage="",
            router_response=_router_response(),
            iteration=1,
        )
        # Set abs_path for the file in prev_result
        prev_result.files[0].abs_path = tmp_path / "src" / "auth" / "jwt.py"

        revised = await writer.revise(
            feature_id=prev_result.feature_id,
            title="Auth",
            existing_result=prev_result,
            review_feedback="Missing error handling.",
        )

        assert revised.iteration == 2


# ---------------------------------------------------------------------------
# ImplementationAgent
# ---------------------------------------------------------------------------


class TestImplementationAgent:

    @pytest.mark.asyncio
    async def test_run_creates_gate_in_phase1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: _ADR_CONTENT)
        monkeypatch.setattr("agents.implementation.agent._read_prd_for_feature", lambda _: _PRD_CONTENT)

        agent, store, fid = _make_agent()
        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.GATE_WAIT
        assert result.gate_id is not None
        store.request_human_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_fails_without_handoff(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        agent, store, fid = _make_agent()
        store.receive_handoff = AsyncMock(return_value=None)

        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.FAILED
        assert "No handoff" in result.error

    @pytest.mark.asyncio
    async def test_run_fails_if_adr_unreadable(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: None)

        agent, store, fid = _make_agent()
        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.FAILED
        assert "ADR" in result.error

    @pytest.mark.asyncio
    async def test_run_registers_implementation_in_graph(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: _ADR_CONTENT)
        monkeypatch.setattr("agents.implementation.agent._read_prd_for_feature", lambda _: _PRD_CONTENT)

        agent, store, fid = _make_agent()
        await agent.run(feature_id=fid)

        store.graph.create_implementation.assert_called_once()
        kwargs = store.graph.create_implementation.call_args[1]
        assert kwargs["feature_id"] == fid

    @pytest.mark.asyncio
    async def test_run_stores_embeddings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: _ADR_CONTENT)
        monkeypatch.setattr("agents.implementation.agent._read_prd_for_feature", lambda _: _PRD_CONTENT)

        agent, store, fid = _make_agent()
        await agent.run(feature_id=fid)

        store.memorize.assert_called_once()
        kwargs = store.memorize.call_args[1]
        assert kwargs["artifact_type"] == "implementation"

    @pytest.mark.asyncio
    async def test_run_creates_handoff_to_testing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: _ADR_CONTENT)
        monkeypatch.setattr("agents.implementation.agent._read_prd_for_feature", lambda _: _PRD_CONTENT)

        agent, store, fid = _make_agent()
        await agent.run(feature_id=fid)

        store.handoff.assert_called_once()
        kwargs = store.handoff.call_args[1]
        assert kwargs["from_agent"] == "implementation"
        assert kwargs["to_agent"] == "testing"

    @pytest.mark.asyncio
    async def test_run_skipped_when_lock_held(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: _ADR_CONTENT)
        monkeypatch.setattr("agents.implementation.agent._read_prd_for_feature", lambda _: _PRD_CONTENT)

        agent, store, fid = _make_agent()

        @asynccontextmanager
        async def _locked_false(*args, **kwargs):
            yield False

        store.locked = _locked_false
        result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_auto_approve_in_phase2(self, tmp_path, monkeypatch):
        """
        Phase 2: always_gate=False, non-breaking, Medium complexity → auto-success.
        """
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: _ADR_CONTENT)
        monkeypatch.setattr("agents.implementation.agent._read_prd_for_feature", lambda _: _PRD_CONTENT)

        agent, store, fid = _make_agent()
        agent._workflow = {
            "stages": {
                "implementation": {
                    "max_regen_iterations": 3,
                    "always_gate": False,
                }
            }
        }

        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.SUCCESS
        store.request_human_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_fails_on_invalid_llm_json(self, tmp_path, monkeypatch):
        """LLM returning non-JSON should result in a FAILED status, not an exception."""
        monkeypatch.setattr("agents.implementation.code_writer.SRC_DIR", tmp_path / "src")
        monkeypatch.setattr("agents.implementation.agent._read_file", lambda _: _ADR_CONTENT)
        monkeypatch.setattr("agents.implementation.agent._read_prd_for_feature", lambda _: _PRD_CONTENT)

        agent, store, fid = _make_agent(router_content="this is not json at all")
        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.FAILED
        assert result.error is not None
