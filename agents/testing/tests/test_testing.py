"""
agents/testing/tests/test_testing.py

Unit tests for the Testing & Security Agent and TestWriter.
All subprocess calls (pytest, bandit) are mocked.

Run: pytest agents/testing/tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agents.base import AgentStatus
from agents.testing.agent import TestingAgent, _load_impl_files
from agents.testing.test_writer import (
    TestWriter,
    TestingResult,
    GeneratedTestFile,
    TestRunResult,
    SASTFinding,
    _build_messages,
    _parse_pytest_output,
    _parse_bandit_output,
    _check_sast_threshold,
    _parse_test_response,
)
from router import RouterResponse


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_VALID_TEST_JSON = {
    "summary": "Tests for JWT authentication module.",
    "files": [
        {
            "path": "tests/generated/test_jwt.py",
            "description": "Tests for jwt.py",
            "content": (
                "import pytest\n"
                "def test_encode():\n"
                "    assert True\n"
            ),
        }
    ],
    "coverage_targets": ["encode", "decode"],
}

_BANDIT_OUTPUT = json.dumps({
    "results": [
        {
            "test_id": "B105",
            "test_name": "hardcoded_password_string",
            "issue_severity": "MEDIUM",
            "issue_confidence": "HIGH",
            "filename": "src/auth/jwt.py",
            "line_number": 12,
            "issue_text": "Possible hardcoded password",
        }
    ],
    "errors": [],
    "metrics": {},
})

_PYTEST_OUTPUT_PASS = "1 passed in 0.12s"
_PYTEST_OUTPUT_FAIL = "2 passed, 1 failed in 0.22s"
_PYTEST_OUTPUT_COVERAGE = (
    "1 passed in 0.12s\n"
    "Name          Stmts   Miss  Cover\n"
    "TOTAL            50     10    80%\n"
)


def _router_response(content: str | None = None) -> RouterResponse:
    return RouterResponse(
        content=content or json.dumps(_VALID_TEST_JSON),
        profile="testing",
        model_used="claude-sonnet-4-6",
        provider="anthropic",
        was_fallback=False,
        input_tokens=250,
        output_tokens=500,
        cost_usd=0.003,
        duration_ms=900,
    )


def _mock_router(content: str | None = None) -> MagicMock:
    r = MagicMock()
    r.complete = AsyncMock(return_value=_router_response(content))
    return r


def _mock_store(feature_id: UUID, handoff: dict | None = None) -> MagicMock:
    store = MagicMock()
    store.receive_handoff = AsyncMock(return_value=handoff or {
        "impl_id": str(uuid4()),
        "title": "User Auth",
        "file_paths": ["src/auth/jwt.py"],
        "is_breaking_change": False,
        "complexity": "Medium",
        "redmine_id": None,
    })
    store.advance_feature = AsyncMock()
    store.graph = MagicMock()
    store.graph.create_test_suite = AsyncMock()
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


def _impl_file(tmp_path: Path) -> tuple[str, Path]:
    p = tmp_path / "src" / "auth" / "jwt.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# jwt.py\ndef encode(p): return p\ndef decode(t): return t\n")
    return str(p), p


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestParsePytestOutput:

    def test_parses_all_passed(self):
        r = _parse_pytest_output(_PYTEST_OUTPUT_PASS, "", 0)
        assert r.passed == 1
        assert r.failed == 0

    def test_parses_mixed(self):
        r = _parse_pytest_output(_PYTEST_OUTPUT_FAIL, "", 1)
        assert r.passed == 2
        assert r.failed == 1

    def test_parses_coverage(self):
        r = _parse_pytest_output(_PYTEST_OUTPUT_COVERAGE, "", 0)
        assert r.coverage_pct == 80.0

    def test_all_passed_true_when_no_failures(self):
        r = _parse_pytest_output(_PYTEST_OUTPUT_PASS, "", 0)
        assert r.all_passed is True

    def test_all_passed_false_when_failures(self):
        r = _parse_pytest_output(_PYTEST_OUTPUT_FAIL, "", 1)
        assert r.all_passed is False


class TestParseBanditOutput:

    def test_parses_finding(self):
        findings = _parse_bandit_output(_BANDIT_OUTPUT)
        assert len(findings) == 1
        assert findings[0].severity == "MEDIUM"
        assert findings[0].test_name == "hardcoded_password_string"

    def test_empty_output_returns_empty(self):
        assert _parse_bandit_output("") == []

    def test_invalid_json_returns_empty(self):
        assert _parse_bandit_output("not json") == []

    def test_no_results_returns_empty(self):
        assert _parse_bandit_output(json.dumps({"results": [], "errors": []})) == []


class TestCheckSASTThreshold:

    def _finding(self, severity: str) -> SASTFinding:
        return SASTFinding(
            severity=severity,
            confidence="HIGH",
            test_id="B001",
            test_name="test",
            filename="src/x.py",
            line_number=1,
            issue_text="issue",
        )

    def test_blocks_on_medium_threshold(self):
        blocked, reason = _check_sast_threshold(
            [self._finding("MEDIUM")], "medium"
        )
        assert blocked is True
        assert "MEDIUM" in reason or "medium" in reason.lower()

    def test_does_not_block_low_below_medium_threshold(self):
        blocked, _ = _check_sast_threshold([self._finding("LOW")], "medium")
        assert blocked is False

    def test_blocks_high_on_medium_threshold(self):
        blocked, _ = _check_sast_threshold([self._finding("HIGH")], "medium")
        assert blocked is True

    def test_empty_findings_no_block(self):
        blocked, _ = _check_sast_threshold([], "medium")
        assert blocked is False

    def test_blocks_only_above_threshold(self):
        findings = [self._finding("LOW"), self._finding("HIGH")]
        blocked, reason = _check_sast_threshold(findings, "high")
        assert blocked is True

    def test_does_not_block_when_all_below(self):
        findings = [self._finding("LOW"), self._finding("LOW")]
        blocked, _ = _check_sast_threshold(findings, "medium")
        assert blocked is False


class TestBuildMessages:

    def test_includes_impl_content(self):
        msgs = _build_messages("Auth", [{"path": "src/a.py", "content": "# code"}], None)
        assert "# code" in msgs[0]["content"]
        assert "Auth" in msgs[0]["content"]

    def test_includes_past_context(self):
        msgs = _build_messages("Auth", [], "past test patterns here")
        assert "past test patterns here" in msgs[0]["content"]


class TestParseTestResponse:

    def test_parses_valid_response(self):
        r = _router_response(json.dumps(_VALID_TEST_JSON))
        files, summary, targets = _parse_test_response(r)
        assert len(files) == 1
        assert files[0].path == "tests/generated/test_jwt.py"
        assert "encode" in targets

    def test_raises_on_invalid_json(self):
        r = _router_response("not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_test_response(r)

    def test_raises_on_missing_files(self):
        r = _router_response(json.dumps({"summary": "x"}))
        with pytest.raises(ValueError, match="missing 'files'"):
            _parse_test_response(r)


class TestLoadImplFiles:

    def test_loads_existing_file(self, tmp_path):
        f = tmp_path / "src" / "a.py"
        f.parent.mkdir(parents=True)
        f.write_text("# code")
        result = _load_impl_files([str(f)])
        assert len(result) == 1
        assert result[0]["content"] == "# code"

    def test_skips_missing_files(self):
        result = _load_impl_files(["/no/such/file.py"])
        assert result == []

    def test_empty_list_returns_empty(self):
        assert _load_impl_files([]) == []


# ---------------------------------------------------------------------------
# TestWriter
# ---------------------------------------------------------------------------


class TestTestWriter:

    @pytest.mark.asyncio
    async def test_write_and_run_generates_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.testing.test_writer.TESTS_DIR", tmp_path / "tests" / "generated")

        # Patch subprocess calls
        with (
            patch("agents.testing.test_writer._run_pytest", new=AsyncMock(
                return_value=TestRunResult(passed=1, failed=0, coverage_pct=90.0, returncode=0)
            )),
            patch("agents.testing.test_writer._run_bandit", new=AsyncMock(return_value=[])),
        ):
            writer = TestWriter(_mock_router())
            result = await writer.write_and_run(
                feature_id=uuid4(),
                title="Auth",
                impl_files=[{"path": "src/auth/jwt.py", "content": "# code"}],
            )

        assert len(result.test_files) == 1
        assert result.test_run.passed == 1
        assert result.sast_blocked is False

    @pytest.mark.asyncio
    async def test_write_and_run_detects_sast_block(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.testing.test_writer.TESTS_DIR", tmp_path / "tests" / "generated")

        finding = SASTFinding("HIGH", "HIGH", "B601", "sql_injection", "src/a.py", 5, "SQL injection")
        with (
            patch("agents.testing.test_writer._run_pytest", new=AsyncMock(
                return_value=TestRunResult(passed=1, returncode=0)
            )),
            patch("agents.testing.test_writer._run_bandit", new=AsyncMock(return_value=[finding])),
        ):
            writer = TestWriter(_mock_router())
            result = await writer.write_and_run(
                feature_id=uuid4(),
                title="Auth",
                impl_files=[{"path": "src/auth/jwt.py", "content": "# code"}],
                block_severity="medium",
            )

        assert result.sast_blocked is True
        assert "sql_injection" in result.sast_block_reason


# ---------------------------------------------------------------------------
# TestingAgent
# ---------------------------------------------------------------------------


class TestTestingAgent:

    def _make_agent(self, fid: UUID | None = None, handoff: dict | None = None, router_content: str | None = None):
        fid = fid or uuid4()
        store = _mock_store(fid, handoff)
        router = _mock_router(router_content)
        agent = TestingAgent(store=store, router=router)
        return agent, store, fid

    @pytest.mark.asyncio
    async def test_run_succeeds_when_tests_pass(self, tmp_path):
        agent, store, fid = self._make_agent()

        with (
            patch("agents.testing.agent._load_impl_files", return_value=[
                {"path": "src/auth/jwt.py", "content": "# code"}
            ]),
            patch("agents.testing.test_writer._run_pytest", new=AsyncMock(
                return_value=TestRunResult(passed=3, failed=0, coverage_pct=85.0, returncode=0)
            )),
            patch("agents.testing.test_writer._run_bandit", new=AsyncMock(return_value=[])),
            patch("agents.testing.test_writer.TESTS_DIR", tmp_path),
        ):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.SUCCESS
        store.graph.create_test_suite.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_fails_on_sast_block(self, tmp_path):
        agent, store, fid = self._make_agent()
        finding = SASTFinding("HIGH", "HIGH", "B601", "sql_injection", "src/a.py", 1, "SQL")

        with (
            patch("agents.testing.agent._load_impl_files", return_value=[
                {"path": "src/auth/jwt.py", "content": "# code"}
            ]),
            patch("agents.testing.test_writer._run_pytest", new=AsyncMock(
                return_value=TestRunResult(passed=1, returncode=0)
            )),
            patch("agents.testing.test_writer._run_bandit", new=AsyncMock(return_value=[finding])),
            patch("agents.testing.test_writer.TESTS_DIR", tmp_path),
        ):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.FAILED
        assert "SAST" in result.error

    @pytest.mark.asyncio
    async def test_run_gates_on_low_coverage(self, tmp_path):
        agent, store, fid = self._make_agent()
        agent._workflow = {"thresholds": {"sast_block_severity": "medium", "min_test_coverage_pct": 80}}

        with (
            patch("agents.testing.agent._load_impl_files", return_value=[
                {"path": "src/auth/jwt.py", "content": "# code"}
            ]),
            patch("agents.testing.test_writer._run_pytest", new=AsyncMock(
                return_value=TestRunResult(passed=1, failed=0, coverage_pct=50.0, returncode=0)
            )),
            patch("agents.testing.test_writer._run_bandit", new=AsyncMock(return_value=[])),
            patch("agents.testing.test_writer.TESTS_DIR", tmp_path),
        ):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.GATE_WAIT
        assert result.gate_id is not None

    @pytest.mark.asyncio
    async def test_run_gates_on_test_failures(self, tmp_path):
        agent, store, fid = self._make_agent()

        with (
            patch("agents.testing.agent._load_impl_files", return_value=[
                {"path": "src/auth/jwt.py", "content": "# code"}
            ]),
            patch("agents.testing.test_writer._run_pytest", new=AsyncMock(
                return_value=TestRunResult(passed=2, failed=1, coverage_pct=90.0, returncode=1)
            )),
            patch("agents.testing.test_writer._run_bandit", new=AsyncMock(return_value=[])),
            patch("agents.testing.test_writer.TESTS_DIR", tmp_path),
        ):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.GATE_WAIT

    @pytest.mark.asyncio
    async def test_run_fails_without_handoff(self, tmp_path):
        agent, store, fid = self._make_agent()
        store.receive_handoff = AsyncMock(return_value=None)
        result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_fails_with_no_impl_files(self, tmp_path):
        agent, store, fid = self._make_agent()
        with patch("agents.testing.agent._load_impl_files", return_value=[]):
            result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_skipped_when_lock_held(self, tmp_path):
        agent, store, fid = self._make_agent()

        @asynccontextmanager
        async def _no_lock(*a, **kw):
            yield False

        store.locked = _no_lock

        with patch("agents.testing.agent._load_impl_files", return_value=[
            {"path": "src/auth/jwt.py", "content": "# code"}
        ]):
            result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_run_creates_handoff_to_review(self, tmp_path):
        agent, store, fid = self._make_agent()

        with (
            patch("agents.testing.agent._load_impl_files", return_value=[
                {"path": "src/auth/jwt.py", "content": "# code"}
            ]),
            patch("agents.testing.test_writer._run_pytest", new=AsyncMock(
                return_value=TestRunResult(passed=1, failed=0, coverage_pct=90.0, returncode=0)
            )),
            patch("agents.testing.test_writer._run_bandit", new=AsyncMock(return_value=[])),
            patch("agents.testing.test_writer.TESTS_DIR", tmp_path),
        ):
            await agent.run(feature_id=fid)

        store.handoff.assert_called_once()
        kwargs = store.handoff.call_args[1]
        assert kwargs["from_agent"] == "testing"
        assert kwargs["to_agent"] == "review"
