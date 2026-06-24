"""
agents/testing/agent.py — L4 Testing & Security Agent

Reads the implementation handoff, generates pytest test files via LLM,
runs them, scans with bandit SAST, then:
  - Blocks pipeline on SAST findings above sast_block_severity threshold
  - Gates if test coverage is below min_test_coverage_pct
  - Otherwise hands off to Review Agent (L5)

Run standalone:
    python -m agents.testing.agent --feature-id <uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from uuid import UUID, uuid4

from agents.base import BaseAgent, AgentResult, AgentStatus
from agents.testing.test_writer import TestWriter, TestingResult
from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)


class TestingAgent(BaseAgent):
    """
    L4 — Testing & Security Agent

    Responsibilities:
      1. Receive implementation handoff
      2. Load implementation file contents from disk
      3. Generate pytest test suite via LLM
      4. Run tests and parse coverage
      5. Run bandit SAST scan
      6. Block on SAST threshold violations
      7. Gate if coverage below minimum
      8. Register test suite in traceability graph
      9. Hand off to Review Agent (L5)
    """

    agent_name = "testing"
    layer = 4

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
    ):
        super().__init__(store=store, router=router)

    async def run(self, feature_id: UUID, **kwargs) -> AgentResult:
        t0 = time.monotonic()
        thresholds = self._workflow.get("thresholds", {})
        block_severity = thresholds.get("sast_block_severity", "medium")
        min_coverage = float(thresholds.get("min_test_coverage_pct", 80))

        # 1. Receive handoff from Implementation Agent
        handoff = await self.store.receive_handoff(
            feature_id=feature_id,
            agent="testing",
        )
        if not handoff:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error="No handoff found from implementation agent",
            )

        title = handoff.get("title", "Untitled")
        file_paths = handoff.get("file_paths", [])
        impl_id = handoff.get("impl_id", "")
        is_breaking = handoff.get("is_breaking_change", False)
        complexity = handoff.get("complexity", "Unknown")
        redmine_id = handoff.get("redmine_id")

        # 2. Load implementation file contents
        impl_files = _load_impl_files(file_paths)
        if not impl_files:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error=f"No readable implementation files found: {file_paths}",
            )

        async with self.store.locked(f"feature:{feature_id}:testing") as acquired:
            if not acquired:
                return AgentResult(
                    status=AgentStatus.SKIPPED,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    error="Testing step already being processed",
                )

            return await self._run_locked(
                feature_id=feature_id,
                title=title,
                impl_files=impl_files,
                impl_id=impl_id,
                is_breaking=is_breaking,
                complexity=complexity,
                redmine_id=redmine_id,
                block_severity=block_severity,
                min_coverage=min_coverage,
                t0=t0,
            )

    async def _run_locked(
        self,
        *,
        feature_id: UUID,
        title: str,
        impl_files: list[dict],
        impl_id: str,
        is_breaking: bool,
        complexity: str,
        redmine_id: int | None,
        block_severity: str,
        min_coverage: float,
        t0: float,
    ) -> AgentResult:

        # Past context for test patterns
        past_context = await self._fetch_past_context(title)

        run_id = await self.begin(
            feature_id=feature_id,
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_summary=(
                f"title={title!r}  files={len(impl_files)}  "
                f"block_severity={block_severity}  min_coverage={min_coverage}"
            ),
        )

        try:
            writer = TestWriter(self._router)
            result: TestingResult = await writer.write_and_run(
                feature_id=feature_id,
                title=title,
                impl_files=impl_files,
                block_severity=block_severity,
                min_coverage_pct=min_coverage,
                existing_context=past_context,
            )

            await self.store.advance_feature(feature_id, "testing")

            # 3. Register test suite in graph
            suite_id = uuid4()
            await self.store.graph.create_test_suite(
                id=suite_id,
                implementation_id=UUID(impl_id) if impl_id else uuid4(),
                test_count=result.test_run.total,
                pass_count=result.test_run.passed,
                fail_count=result.test_run.failed,
                coverage_pct=result.test_run.coverage_pct,
            )

            # 4. Embed test files
            combined_tests = "\n\n".join(
                f"# === {f.path} ===\n{f.content}" for f in result.test_files
            )
            if combined_tests:
                await self.store.memorize(
                    combined_tests,
                    feature_id=feature_id,
                    artifact_type="test_suite",
                    artifact_id=str(suite_id),
                )

            duration_ms = int((time.monotonic() - t0) * 1000)
            cost = result.router_response.cost_usd

            # 5. SAST block check — hard stop, no gate, must fix
            if result.sast_blocked:
                await self.store.record_decision(
                    run_id=run_id,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    decision_type="sast_blocked",
                    summary=f"SAST block: {result.sast_block_reason}",
                    rationale=f"Threshold: {block_severity}",
                    outcome="blocked",
                )
                return await self.fail(
                    run_id=run_id,
                    feature_id=feature_id,
                    error=f"SAST blocked: {result.sast_block_reason}",
                    cost_usd=cost,
                    duration_ms=duration_ms,
                )

            # 6. Coverage gate
            coverage_ok = result.test_run.coverage_pct >= min_coverage
            tests_ok = result.test_run.all_passed or result.test_run.total == 0

            gate_needed = not tests_ok or not coverage_ok

            await self.store.record_decision(
                run_id=run_id,
                feature_id=feature_id,
                agent=self.agent_name,
                decision_type="testing_complete",
                summary=(
                    f"Tests: {result.test_run.passed}/{result.test_run.total} passed, "
                    f"coverage={result.test_run.coverage_pct:.1f}%  "
                    f"sast={len(result.sast_findings)} findings"
                ),
                rationale=(
                    f"coverage_ok={coverage_ok}  tests_ok={tests_ok}  "
                    f"sast_blocked={result.sast_blocked}"
                ),
                outcome="gate_required" if gate_needed else "auto_approved",
            )

            # 7. Hand off to Review Agent
            await self.store.handoff(
                feature_id=feature_id,
                from_agent="testing",
                to_agent="review",
                payload={
                    "suite_id": str(suite_id),
                    "impl_id": impl_id,
                    "title": title,
                    "file_paths": [f["path"] for f in impl_files],
                    "test_file_paths": [f.path for f in result.test_files],
                    "passed": result.test_run.passed,
                    "failed": result.test_run.failed,
                    "coverage_pct": result.test_run.coverage_pct,
                    "sast_findings": len(result.sast_findings),
                    "is_breaking_change": is_breaking,
                    "complexity": complexity,
                    "redmine_id": redmine_id,
                },
            )

            if gate_needed:
                reasons = []
                if not tests_ok:
                    reasons.append(f"{result.test_run.failed} test(s) failed")
                if not coverage_ok:
                    reasons.append(
                        f"coverage {result.test_run.coverage_pct:.1f}% < {min_coverage:.0f}% minimum"
                    )
                return await self.gate(
                    run_id=run_id,
                    feature_id=feature_id,
                    gate_type="test_quality_review",
                    message=(
                        f"Testing gate for '{title}': {'; '.join(reasons)}. "
                        "Review and decide whether to proceed to code review."
                    ),
                    trigger_reason="; ".join(reasons),
                    payload={
                        "suite_id": str(suite_id),
                        "passed": result.test_run.passed,
                        "failed": result.test_run.failed,
                        "coverage_pct": result.test_run.coverage_pct,
                        "sast_findings": len(result.sast_findings),
                    },
                    cost_usd=cost,
                    duration_ms=duration_ms,
                )

            return await self.succeed(
                run_id=run_id,
                feature_id=feature_id,
                output={
                    "suite_id": str(suite_id),
                    "passed": result.test_run.passed,
                    "coverage_pct": result.test_run.coverage_pct,
                    "sast_findings": len(result.sast_findings),
                },
                cost_usd=cost,
                duration_ms=duration_ms,
                output_summary=(
                    f"Tests passed: {result.test_run.passed}/{result.test_run.total}, "
                    f"coverage={result.test_run.coverage_pct:.1f}%"
                ),
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("TestingAgent failed for feature %s", feature_id)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def _fetch_past_context(self, title: str) -> str | None:
        try:
            hits = await self.store.remember(title, limit=3, artifact_type="test_suite")
            if not hits:
                return None
            return "\n\n".join(
                f"[Past test excerpt, similarity={h['similarity']:.2f}]\n{h['content']}"
                for h in hits
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_impl_files(file_paths: list[str]) -> list[dict]:
    """Read implementation files from disk. Returns list of {path, content} dicts."""
    files = []
    for p in file_paths:
        path = Path(p)
        if path.exists():
            try:
                files.append({"path": p, "content": path.read_text(encoding="utf-8")})
            except Exception as exc:
                logger.warning("Could not read %s: %s", p, exc)
    return files


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description="AI SDLC — Testing Agent (L4)")
    parser.add_argument("--feature-id", "-f", required=True, type=UUID)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with TestingAgent() as agent:
        result = await agent.run(feature_id=args.feature_id)

    print(f"\n{'═' * 60}")
    print(f"  Status:     {result.status.value}")
    print(f"  Feature ID: {result.feature_id}")
    if result.gate_id:
        print(f"  Gate ID:    {result.gate_id}")
    elif result.error:
        print(f"  Error:      {result.error}")
    print(f"  Cost:       ${result.cost_usd:.6f}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(_main())
