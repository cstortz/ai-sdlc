"""
agents/testing/test_writer.py — Test generation + SAST scanning.

Takes the generated implementation files and:
  1. Calls LLM (testing profile) to write pytest test files
  2. Runs the tests via subprocess and parses coverage
  3. Runs bandit SAST scan and parses findings
  4. Returns a structured TestingResult
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from router import ModelRouter, RouterResponse

logger = logging.getLogger(__name__)

TESTS_DIR = Path(__file__).parent.parent.parent / "tests" / "generated"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior QA engineer in an AI-powered SDLC pipeline.
Your job is to write comprehensive pytest unit tests for the provided implementation files.

Output a JSON object ONLY — no preamble, no explanation, no code fences.

Schema:
{
  "summary": "One-sentence description of the test suite.",
  "files": [
    {
      "path": "tests/generated/test_{module}.py",
      "description": "What this test file covers.",
      "content": "Full pytest file content."
    }
  ],
  "coverage_targets": ["List of functions/classes that MUST be tested."]
}

Rules:
- All test paths must start with "tests/generated/".
- Use pytest and pytest-asyncio for async tests.
- Mock all external dependencies (DB, Redis, APIs) using unittest.mock.
- Include happy-path, edge-case, and error-path tests for every public function.
- Use parametrize for data-driven tests where appropriate.
- Tests must be fully self-contained — no real DB connections, no real API calls.
- Each test function must have a clear docstring stating what it verifies.
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SASTFinding:
    severity: str        # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    confidence: str      # "LOW" | "MEDIUM" | "HIGH"
    test_id: str         # e.g. "B105"
    test_name: str       # e.g. "hardcoded_password_string"
    filename: str
    line_number: int
    issue_text: str


@dataclass
class TestRunResult:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    coverage_pct: float = 0.0
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.total > 0


@dataclass
class GeneratedTestFile:
    path: str
    description: str
    content: str
    abs_path: Path = field(init=False)

    def __post_init__(self):
        self.abs_path = Path(__file__).parent.parent.parent / self.path


@dataclass
class TestingResult:
    """Complete output of the testing step."""
    feature_id: UUID
    title: str
    test_files: list[GeneratedTestFile]
    test_run: TestRunResult
    sast_findings: list[SASTFinding]
    sast_blocked: bool           # True if a finding exceeded the block threshold
    sast_block_reason: str       # Human-readable reason for block
    router_response: RouterResponse
    summary: str
    coverage_targets: list[str]


# ---------------------------------------------------------------------------
# Test writer + runner
# ---------------------------------------------------------------------------


class TestWriter:
    """
    Generates pytest test files from implementation code, then runs them.

    Usage:
        writer = TestWriter(router)
        result = await writer.write_and_run(feature_id, title, impl_files, block_severity)
    """

    def __init__(self, router: ModelRouter):
        self._router = router

    async def write_and_run(
        self,
        feature_id: UUID,
        title: str,
        impl_files: list[dict],          # [{"path": ..., "content": ...}]
        block_severity: str = "medium",  # from workflow.yaml sast_block_severity
        min_coverage_pct: float = 80.0,  # from workflow.yaml
        existing_context: str | None = None,
    ) -> TestingResult:
        """
        Generate test files, write them to disk, run pytest + bandit, return results.
        """
        # 1. Generate tests
        messages = _build_messages(title, impl_files, existing_context)
        logger.info("Generating tests for feature %s: %r", feature_id, title)

        response = await self._router.complete(
            profile="testing",
            messages=messages,
            system=SYSTEM_PROMPT,
        )

        test_files, summary, coverage_targets = _parse_test_response(response)
        _write_test_files(test_files)

        # 2. Run pytest
        test_run = await _run_pytest(test_files)

        # 3. Run bandit SAST on implementation files
        impl_paths = [f["path"] for f in impl_files]
        sast_findings = await _run_bandit(impl_paths)

        # 4. Check SAST block threshold
        blocked, block_reason = _check_sast_threshold(sast_findings, block_severity)

        logger.info(
            "Testing complete: %d passed, %d failed, coverage=%.1f%%, sast=%d findings, blocked=%s",
            test_run.passed, test_run.failed, test_run.coverage_pct,
            len(sast_findings), blocked,
        )

        return TestingResult(
            feature_id=feature_id,
            title=title,
            test_files=test_files,
            test_run=test_run,
            sast_findings=sast_findings,
            sast_blocked=blocked,
            sast_block_reason=block_reason,
            router_response=response,
            summary=summary,
            coverage_targets=coverage_targets,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (testable)
# ---------------------------------------------------------------------------


def _build_messages(
    title: str,
    impl_files: list[dict],
    existing_context: str | None,
) -> list[dict]:
    files_text = "\n\n".join(
        f"=== {f['path']} ===\n{f['content']}" for f in impl_files
    )
    user_content = f"Feature: {title}\n\nImplementation files:\n\n{files_text}"
    if existing_context:
        user_content = f"Relevant past tests:\n{existing_context}\n\n---\n\n{user_content}"
    return [{"role": "user", "content": user_content}]


def _parse_test_response(
    response: RouterResponse,
) -> tuple[list[GeneratedTestFile], str, list[str]]:
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for tests: {exc}") from exc

    if "files" not in data:
        raise ValueError("LLM test response missing 'files' list")

    files = [
        GeneratedTestFile(
            path=f["path"],
            description=f.get("description", ""),
            content=f["content"],
        )
        for f in data["files"]
        if "path" in f and "content" in f
    ]
    return files, data.get("summary", ""), data.get("coverage_targets", [])


def _write_test_files(files: list[GeneratedTestFile]) -> None:
    for f in files:
        f.abs_path.parent.mkdir(parents=True, exist_ok=True)
        f.abs_path.write_text(f.content, encoding="utf-8")
        logger.debug("Wrote test file %s", f.abs_path)


async def _run_pytest(
    test_files: list[GeneratedTestFile],
    timeout: int = 120,
) -> TestRunResult:
    """
    Run pytest on the generated test files.
    Returns a TestRunResult regardless of exit code.
    """
    if not test_files:
        return TestRunResult()

    paths = [str(f.abs_path) for f in test_files]
    cmd = [
        sys.executable, "-m", "pytest",
        "--asyncio-mode=auto",
        "--tb=short",
        "-q",
        f"--cov=src",
        "--cov-report=term-missing",
        "--no-header",
        *paths,
    ]

    try:
        loop = asyncio.get_event_loop()
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(__file__).parent.parent.parent),
            ),
        )
        return _parse_pytest_output(proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("pytest timed out after %ds", timeout)
        return TestRunResult(returncode=-1, stderr=f"Timed out after {timeout}s")
    except Exception as exc:
        logger.warning("Failed to run pytest: %s", exc)
        return TestRunResult(returncode=-1, stderr=str(exc))


def _parse_pytest_output(stdout: str, stderr: str, returncode: int) -> TestRunResult:
    """Parse pytest's short output to extract pass/fail counts and coverage."""
    result = TestRunResult(stdout=stdout, stderr=stderr, returncode=returncode)

    # e.g. "5 passed, 1 failed in 0.42s" or "3 passed in 0.12s"
    summary_match = re.search(
        r"(\d+) passed(?:,\s*(\d+) failed)?(?:,\s*(\d+) error)?",
        stdout,
    )
    if summary_match:
        result.passed = int(summary_match.group(1) or 0)
        result.failed = int(summary_match.group(2) or 0)
        result.errors = int(summary_match.group(3) or 0)

    # e.g. "TOTAL    120    24    80%"
    cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", stdout)
    if cov_match:
        result.coverage_pct = float(cov_match.group(1))

    return result


async def _run_bandit(
    file_paths: list[str],
    timeout: int = 60,
) -> list[SASTFinding]:
    """
    Run bandit SAST scan on the given file paths.
    Returns an empty list if bandit is not installed or scan fails.
    """
    if not file_paths:
        return []

    # Only scan files that exist
    existing = [p for p in file_paths if Path(p).exists()]
    if not existing:
        return []

    cmd = [sys.executable, "-m", "bandit", "-r", "-f", "json", *existing]
    try:
        loop = asyncio.get_event_loop()
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            ),
        )
        return _parse_bandit_output(proc.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as exc:
        logger.warning("bandit scan failed (skipping): %s", exc)
        return []


def _parse_bandit_output(output: str) -> list[SASTFinding]:
    """Parse bandit's JSON output into SASTFinding list."""
    if not output.strip():
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []

    findings = []
    for r in data.get("results", []):
        findings.append(SASTFinding(
            severity=r.get("issue_severity", "LOW").upper(),
            confidence=r.get("issue_confidence", "LOW").upper(),
            test_id=r.get("test_id", ""),
            test_name=r.get("test_name", ""),
            filename=r.get("filename", ""),
            line_number=r.get("line_number", 0),
            issue_text=r.get("issue_text", ""),
        ))
    return findings


_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _check_sast_threshold(
    findings: list[SASTFinding],
    block_severity: str,
) -> tuple[bool, str]:
    """
    Returns (blocked, reason).
    Blocked if any finding's severity >= block_severity.
    """
    threshold = _SEVERITY_RANK.get(block_severity.upper(), 2)
    blocking = [
        f for f in findings
        if _SEVERITY_RANK.get(f.severity, 1) >= threshold
    ]
    if blocking:
        details = "; ".join(
            f"{f.test_name} ({f.severity}) in {f.filename}:{f.line_number}"
            for f in blocking[:5]
        )
        return True, f"{len(blocking)} SAST finding(s) at or above {block_severity.upper()}: {details}"
    return False, ""
