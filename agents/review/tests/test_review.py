"""
agents/review/tests/test_review.py

Unit tests for the Review Agent and CodeReviewer.

Run: pytest agents/review/tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agents.base import AgentStatus
from agents.review.agent import ReviewAgent, _decide_outcome, _load_files
from agents.review.reviewer import (
    CodeReviewer,
    ReviewResult,
    PRResult,
    _build_review_messages,
    _parse_review_response,
    _null_router_response,
)
from router import RouterResponse


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_REVIEW_JSON_APPROVED = {
    "score": 88,
    "summary": "Well-structured implementation with good error handling.",
    "issues": [
        {
            "severity": "low",
            "file": "src/auth/jwt.py",
            "line": 42,
            "description": "Variable name `x` is not descriptive.",
            "suggestion": "Rename to `token_expiry`.",
        }
    ],
    "strengths": ["Good type hints", "Comprehensive error handling"],
    "must_fix": [],
    "approved": True,
}

_REVIEW_JSON_REJECTED = {
    "score": 55,
    "summary": "Several security issues and missing error handling.",
    "issues": [
        {
            "severity": "critical",
            "file": "src/auth/jwt.py",
            "line": 10,
            "description": "Hardcoded secret key.",
            "suggestion": "Use environment variable.",
        }
    ],
    "strengths": [],
    "must_fix": ["Remove hardcoded secret key"],
    "approved": False,
}

_REVIEW_JSON_GATE = {
    "score": 75,
    "summary": "Acceptable but has some medium issues.",
    "issues": [],
    "strengths": ["Tests present"],
    "must_fix": [],
    "approved": True,
}

_PR_BODY = "## Summary\nAdds JWT auth.\n\n## Changes\n- src/auth/jwt.py\n"


def _router_response(content: str | None = None, profile: str = "code_review") -> RouterResponse:
    return RouterResponse(
        content=content or json.dumps(_REVIEW_JSON_APPROVED),
        profile=profile,
        model_used="claude-sonnet-4-6",
        provider="anthropic",
        was_fallback=False,
        input_tokens=400,
        output_tokens=300,
        cost_usd=0.003,
        duration_ms=700,
    )


def _mock_router(review_content: str | None = None, pr_content: str = _PR_BODY) -> MagicMock:
    router = MagicMock()

    async def _complete(*args, **kwargs):
        profile = kwargs.get("profile", "")
        if profile == "pr_description":
            return _router_response(pr_content, "pr_description")
        return _router_response(review_content)

    router.complete = AsyncMock(side_effect=_complete)
    return router


def _mock_store(fid: UUID, handoff: dict | None = None) -> MagicMock:
    store = MagicMock()
    store.receive_handoff = AsyncMock(return_value=handoff or {
        "impl_id": str(uuid4()),
        "suite_id": str(uuid4()),
        "title": "User Auth",
        "file_paths": ["src/auth/jwt.py"],
        "passed": 5,
        "failed": 0,
        "coverage_pct": 85.0,
        "sast_findings": 0,
        "is_breaking_change": False,
        "complexity": "Medium",
        "redmine_id": None,
    })
    store.advance_feature = AsyncMock()
    store.graph = MagicMock()
    store.graph.update_node_status = AsyncMock()
    store.memorize = AsyncMock(return_value=1)
    store.begin_run = AsyncMock(return_value=uuid4())
    store.end_run = AsyncMock()
    store.request_human_approval = AsyncMock(return_value=uuid4())
    store.record_decision = AsyncMock(return_value=uuid4())
    store.handoff = AsyncMock(return_value=uuid4())
    store.remember = AsyncMock(return_value=[])
    store.cache = MagicMock()
    store.cache.client = MagicMock()
    store.cache.client.publish = AsyncMock(return_value=1)

    @asynccontextmanager
    async def _locked(*a, **kw):
        yield True

    store.locked = _locked
    return store


def _make_agent(
    fid: UUID | None = None,
    handoff: dict | None = None,
    review_json: dict | None = None,
) -> tuple[ReviewAgent, MagicMock, UUID]:
    fid = fid or uuid4()
    store = _mock_store(fid, handoff)
    router = _mock_router(json.dumps(review_json) if review_json else None)
    agent = ReviewAgent(store=store, router=router)
    return agent, store, fid


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestDecideOutcome:

    def test_auto_merge_above_threshold(self):
        assert _decide_outcome(90, 85, 70) == "auto_merge"

    def test_gate_between_thresholds(self):
        assert _decide_outcome(75, 85, 70) == "gate"

    def test_regen_below_regen_threshold(self):
        assert _decide_outcome(60, 85, 70) == "regen"

    def test_exactly_at_auto_merge(self):
        assert _decide_outcome(85, 85, 70) == "auto_merge"

    def test_exactly_at_regen(self):
        assert _decide_outcome(70, 85, 70) == "gate"

    def test_just_below_regen(self):
        assert _decide_outcome(69.9, 85, 70) == "regen"


class TestLoadFiles:

    def test_loads_existing_file(self, tmp_path):
        f = tmp_path / "src" / "a.py"
        f.parent.mkdir(parents=True)
        f.write_text("# code")
        result = _load_files([str(f)])
        assert len(result) == 1

    def test_skips_missing(self):
        assert _load_files(["/no/such/file.py"]) == []


# ---------------------------------------------------------------------------
# CodeReviewer
# ---------------------------------------------------------------------------


class TestParseReviewResponse:

    def test_parses_approved_review(self):
        r = _router_response(json.dumps(_REVIEW_JSON_APPROVED))
        result = _parse_review_response(r)
        assert result.score == 88
        assert result.approved is True
        assert len(result.issues) == 1
        assert result.issues[0].severity == "low"

    def test_parses_rejected_review(self):
        r = _router_response(json.dumps(_REVIEW_JSON_REJECTED))
        result = _parse_review_response(r)
        assert result.score == 55
        assert result.approved is False
        assert len(result.must_fix) == 1

    def test_raises_on_invalid_json(self):
        r = _router_response("not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_review_response(r)

    def test_parses_json_inside_code_fence(self):
        fenced = f"```json\n{json.dumps(_REVIEW_JSON_APPROVED)}\n```"
        r = _router_response(fenced)
        result = _parse_review_response(r)
        assert result.score == 88


class TestBuildReviewMessages:

    def test_includes_file_contents(self):
        msgs = _build_review_messages(
            "Auth",
            [{"path": "src/a.py", "content": "def foo(): pass"}],
            {"passed": 3, "failed": 0, "coverage_pct": 80.0, "sast_findings": 0},
            None, None,
        )
        assert "def foo(): pass" in msgs[0]["content"]

    def test_includes_test_summary(self):
        msgs = _build_review_messages(
            "Auth",
            [],
            {"passed": 3, "failed": 1, "coverage_pct": 75.5, "sast_findings": 2},
            None, None,
        )
        body = msgs[0]["content"]
        assert "75.5" in body
        assert "sast_findings=2" in body.lower() or "2" in body

    def test_includes_adr_summary(self):
        msgs = _build_review_messages("T", [], {}, "Use JWT for auth", None)
        assert "JWT" in msgs[0]["content"]

    def test_includes_past_context(self):
        msgs = _build_review_messages("T", [], {}, None, "past review patterns")
        assert "past review patterns" in msgs[0]["content"]


class TestCodeReviewer:

    @pytest.mark.asyncio
    async def test_review_returns_result(self):
        router = _mock_router()
        reviewer = CodeReviewer(router)
        result = await reviewer.review(
            feature_id=uuid4(),
            title="Auth",
            impl_files=[{"path": "src/a.py", "content": "# code"}],
            test_summary={"passed": 3, "failed": 0, "coverage_pct": 80.0, "sast_findings": 0},
        )
        assert result.score == 88
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_generate_pr_description_calls_router(self):
        router = _mock_router()
        reviewer = CodeReviewer(router)
        review = ReviewResult(
            score=88,
            summary="Good",
            issues=[],
            strengths=["Clean code"],
            must_fix=[],
            approved=True,
            router_response=_router_response(),
        )
        body = await reviewer.generate_pr_description(
            feature_id=uuid4(),
            title="Auth",
            review=review,
            test_summary={"passed": 3, "failed": 0, "coverage_pct": 80.0, "sast_findings": 0},
            is_breaking=False,
            file_paths=["src/auth/jwt.py"],
        )
        assert len(body) > 0

    @pytest.mark.asyncio
    async def test_create_github_pr_skips_without_token(self):
        reviewer = CodeReviewer(MagicMock())
        result = await reviewer.create_github_pr(
            repo_name="owner/repo",
            title="feat: auth",
            body="PR body",
            branch="feature/abc",
            github_token=None,
        )
        assert result.pr_number is None


# ---------------------------------------------------------------------------
# ReviewAgent
# ---------------------------------------------------------------------------


class TestReviewAgent:

    @pytest.mark.asyncio
    async def test_auto_merge_on_high_score(self, tmp_path):
        agent, store, fid = _make_agent(review_json=_REVIEW_JSON_APPROVED)  # score=88
        agent._workflow = {"thresholds": {
            "code_review_score_auto_merge": 0.85,
            "code_review_score_regen": 0.70,
        }}

        with patch("agents.review.agent._load_files", return_value=[
            {"path": "src/auth/jwt.py", "content": "# code"}
        ]):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.SUCCESS
        store.request_human_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_regen_on_low_score(self, tmp_path):
        agent, store, fid = _make_agent(review_json=_REVIEW_JSON_REJECTED)  # score=55
        agent._workflow = {"thresholds": {
            "code_review_score_auto_merge": 0.85,
            "code_review_score_regen": 0.70,
        }}

        with patch("agents.review.agent._load_files", return_value=[
            {"path": "src/auth/jwt.py", "content": "# code"}
        ]):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.FAILED
        assert "regen threshold" in result.error.lower() or "below" in result.error.lower()

    @pytest.mark.asyncio
    async def test_gate_on_medium_score(self, tmp_path):
        agent, store, fid = _make_agent(review_json=_REVIEW_JSON_GATE)  # score=75
        agent._workflow = {"thresholds": {
            "code_review_score_auto_merge": 0.85,
            "code_review_score_regen": 0.70,
        }}

        with patch("agents.review.agent._load_files", return_value=[
            {"path": "src/auth/jwt.py", "content": "# code"}
        ]):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.GATE_WAIT
        assert result.gate_id is not None

    @pytest.mark.asyncio
    async def test_run_fails_without_handoff(self):
        agent, store, fid = _make_agent()
        store.receive_handoff = AsyncMock(return_value=None)
        result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_fails_with_no_files(self):
        agent, store, fid = _make_agent()
        with patch("agents.review.agent._load_files", return_value=[]):
            result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_creates_handoff_to_deploy(self):
        agent, store, fid = _make_agent(review_json=_REVIEW_JSON_APPROVED)
        agent._workflow = {"thresholds": {
            "code_review_score_auto_merge": 0.85,
            "code_review_score_regen": 0.70,
        }}

        with patch("agents.review.agent._load_files", return_value=[
            {"path": "src/auth/jwt.py", "content": "# code"}
        ]):
            await agent.run(feature_id=fid)

        store.handoff.assert_called_once()
        kwargs = store.handoff.call_args[1]
        assert kwargs["from_agent"] == "review"
        assert kwargs["to_agent"] == "deploy"

    @pytest.mark.asyncio
    async def test_run_skipped_when_lock_held(self):
        agent, store, fid = _make_agent()

        @asynccontextmanager
        async def _no(*a, **kw):
            yield False

        store.locked = _no

        with patch("agents.review.agent._load_files", return_value=[
            {"path": "src/auth/jwt.py", "content": "# code"}
        ]):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.SKIPPED
