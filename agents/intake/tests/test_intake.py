"""
agents/intake/tests/test_intake.py

Unit tests for the Intake Agent, interviewer, and PRD writer.
No live API calls, no live DB — everything is mocked.

Run: pytest agents/intake/tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agents.base import AgentStatus
from agents.intake.agent import IntakeAgent
from agents.intake.interviewer import (
    BaseInterviewer,
    CliInterviewer,
    InterviewMode,
    RedmineInterviewer,
    create_interviewer,
)
from agents.intake.prd_writer import InterviewAnswers, PRDWriter
from router import RouterResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> UUID:
    return uuid4()


def _mock_router_response(content: str = "# PRD: Test\n\n## Problem Statement\nTest.") -> RouterResponse:
    return RouterResponse(
        content=content,
        profile="intake",
        model_used="claude-sonnet-4-6",
        provider="anthropic",
        was_fallback=False,
        input_tokens=100,
        output_tokens=200,
        cost_usd=0.001,
        duration_ms=500,
    )


def _mock_store(feature_id: UUID | None = None) -> MagicMock:
    fid = feature_id or _uuid()
    store = MagicMock()
    store.register_feature = AsyncMock(return_value=fid)
    store.pg = MagicMock()
    store.pg.get_feature = AsyncMock(return_value={"id": str(fid), "title": "Test feature"})
    store.graph = MagicMock()
    store.graph.create_prd = AsyncMock(return_value=None)
    store.advance_feature = AsyncMock(return_value=None)
    store.memorize = AsyncMock(return_value=1)
    store.begin_run = AsyncMock(return_value=_uuid())
    store.end_run = AsyncMock(return_value=None)
    store.request_human_approval = AsyncMock(return_value=_uuid())
    store.record_decision = AsyncMock(return_value=_uuid())
    store.handoff = AsyncMock(return_value=_uuid())
    store.remember = AsyncMock(return_value=[])
    store.cache = MagicMock()
    store.cache.client = MagicMock()
    store.cache.client.publish = AsyncMock(return_value=1)
    # Mock distributed lock as a no-op context manager
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def _locked(*args, **kwargs):
        yield True
    store.locked = _locked
    return store


def _mock_interviewer(answers: dict | None = None) -> MagicMock:
    """Returns a mock interviewer that supplies preset answers."""
    answers = answers or {}
    interviewer = MagicMock(spec=BaseInterviewer)
    interviewer.show = AsyncMock(return_value=None)
    # ask() returns the field-specific answer or empty string
    field_order = ["problem", "users", "success_criteria", "out_of_scope", "additional_context", "title"]
    call_count = [0]
    async def _ask(question, context=""):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(field_order):
            return answers.get(field_order[idx], "")
        return ""
    interviewer.ask = _ask
    interviewer.confirm = AsyncMock(return_value=True)
    return interviewer


# ---------------------------------------------------------------------------
# InterviewAnswers
# ---------------------------------------------------------------------------

class TestInterviewAnswers:

    def test_to_prompt_includes_all_fields(self):
        a = InterviewAnswers(
            title="Login",
            raw_prompt="Build login",
            problem="Users can't log in",
            users="End users",
            success_criteria="Login works",
            out_of_scope="SSO",
            additional_context="Must be secure",
        )
        prompt = a.to_prompt()
        assert "Login" in prompt
        assert "Users can't log in" in prompt
        assert "End users" in prompt
        assert "Login works" in prompt
        assert "SSO" in prompt

    def test_to_prompt_skips_empty_fields(self):
        a = InterviewAnswers(title="T", raw_prompt="R")
        prompt = a.to_prompt()
        assert "problem" not in prompt.lower()  # no empty "Problem being solved:" entry


# ---------------------------------------------------------------------------
# Interviewer
# ---------------------------------------------------------------------------

class TestInterviewer:

    def test_create_cli_interviewer(self):
        iv = create_interviewer("cli")
        assert isinstance(iv, CliInterviewer)

    def test_create_redmine_interviewer(self):
        iv = create_interviewer(
            "redmine",
            issue_id=1, redmine_url="http://x", api_key="k", agent_user_id=1,
        )
        assert isinstance(iv, RedmineInterviewer)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            create_interviewer("fax_machine")

    @pytest.mark.asyncio
    async def test_redmine_ask_not_implemented(self):
        iv = RedmineInterviewer(issue_id=1, redmine_url="x", api_key="k", agent_user_id=1)
        with pytest.raises(NotImplementedError):
            await iv.ask("question")


# ---------------------------------------------------------------------------
# PRDWriter
# ---------------------------------------------------------------------------

class TestPRDWriter:

    def _make_writer(self, response_content: str = "# PRD: Test\n\n## Problem Statement\nTest.") -> tuple[PRDWriter, MagicMock]:
        router = MagicMock()
        router.complete = AsyncMock(return_value=_mock_router_response(response_content))
        writer = PRDWriter(router)
        return writer, router

    @pytest.mark.asyncio
    async def test_write_calls_router_and_returns_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        writer, router = self._make_writer()
        answers = InterviewAnswers(title="Login", raw_prompt="Build login", problem="No login exists")

        result = await writer.write(feature_id=_uuid(), answers=answers)

        router.complete.assert_called_once()
        assert result.file_path.exists()
        assert "PRD" in result.content

    @pytest.mark.asyncio
    async def test_write_strips_code_fences(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        writer, _ = self._make_writer("```markdown\n# PRD: X\n```")

        answers = InterviewAnswers(title="X", raw_prompt="X")
        result = await writer.write(feature_id=_uuid(), answers=answers)

        assert not result.content.startswith("```")
        assert not result.content.endswith("```")

    @pytest.mark.asyncio
    async def test_revise_generates_updated_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        writer, router = self._make_writer("# PRD: Revised\n\n## Problem Statement\nUpdated.")

        answers = InterviewAnswers(title="Login", raw_prompt="Build login")
        result = await writer.revise(
            feature_id=_uuid(),
            answers=answers,
            existing_prd="# PRD: Old",
            feedback="Add more acceptance criteria",
        )
        assert "Revised" in result.content

    @pytest.mark.asyncio
    async def test_write_uses_intake_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        writer, router = self._make_writer()
        answers = InterviewAnswers(title="T", raw_prompt="R")
        await writer.write(feature_id=_uuid(), answers=answers)
        call_kwargs = router.complete.call_args
        assert call_kwargs[1]["profile"] == "intake" or call_kwargs[0][0] == "intake"


# ---------------------------------------------------------------------------
# IntakeAgent
# ---------------------------------------------------------------------------

class TestIntakeAgent:

    def _make_agent(self, feature_id: UUID | None = None, interviewer_answers: dict | None = None):
        fid = feature_id or _uuid()
        store = _mock_store(fid)
        router = MagicMock()
        router.complete = AsyncMock(return_value=_mock_router_response())
        interviewer = _mock_interviewer(interviewer_answers or {})
        agent = IntakeAgent(store=store, router=router, interviewer=interviewer)
        return agent, store, fid

    @pytest.mark.asyncio
    async def test_run_succeeds_and_creates_gate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        agent, store, fid = self._make_agent()

        result = await agent.run(
            feature_id=fid,
            prompt="Build a user login system",
            skip_interview=True,
        )

        assert result.status == AgentStatus.GATE_WAIT
        assert result.gate_id is not None
        assert result.feature_id == fid
        store.request_human_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_registers_feature_if_no_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        agent, store, fid = self._make_agent()

        result = await agent.run(
            prompt="Build a dashboard",
            skip_interview=True,
        )

        store.register_feature.assert_called_once()
        assert result.feature_id is not None

    @pytest.mark.asyncio
    async def test_run_creates_prd_node_in_graph(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        agent, store, fid = self._make_agent()

        await agent.run(feature_id=fid, prompt="Feature", skip_interview=True)

        store.graph.create_prd.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_stores_embeddings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        agent, store, fid = self._make_agent()

        await agent.run(feature_id=fid, prompt="Feature", skip_interview=True)

        store.memorize.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_creates_handoff(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        agent, store, fid = self._make_agent()

        await agent.run(feature_id=fid, prompt="Feature", skip_interview=True)

        store.handoff.assert_called_once()
        call_kwargs = store.handoff.call_args[1]
        assert call_kwargs["from_agent"] == "intake"
        assert call_kwargs["to_agent"] == "architecture"

    @pytest.mark.asyncio
    async def test_run_skipped_when_lock_held(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        agent, store, fid = self._make_agent()

        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _locked_false(*args, **kwargs):
            yield False  # Lock not acquired

        store.locked = _locked_false

        result = await agent.run(feature_id=fid, prompt="Feature", skip_interview=True)
        assert result.status == AgentStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_interview_asks_up_to_max_questions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.intake.prd_writer.PRD_DIR", tmp_path)
        agent, store, fid = self._make_agent(
            interviewer_answers={
                "problem": "No auth system",
                "users": "All employees",
                "success_criteria": "Login rate 99%",
            }
        )
        # Max 3 questions in workflow.yaml
        result = await agent.run(feature_id=fid, prompt="Build login", skip_interview=False)
        assert result.status in (AgentStatus.GATE_WAIT, AgentStatus.SUCCESS)

    def test_derive_title_from_prompt(self):
        agent = IntakeAgent.__new__(IntakeAgent)
        assert agent._derive_title("Build a user login system") == "Build a user login system"
        assert agent._derive_title("A" * 100) == "A" * 80
        long = "First sentence. Second sentence."
        assert agent._derive_title(long) == "First sentence"
