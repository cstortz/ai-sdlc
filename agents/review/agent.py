"""
agents/review/agent.py — L5 Code Review Agent

Reads the testing handoff, runs LLM code review, then:
  - score >= auto_merge threshold → creates GitHub PR, hands off to Deploy
  - score < regen threshold → triggers Implementation Agent re-generation
  - otherwise → creates human gate for review decision

Also generates the PR description and creates the GitHub PR (even when gating).

Run standalone:
    python -m agents.review.agent --feature-id <uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path
from uuid import UUID, uuid4

from agents.base import BaseAgent, AgentResult, AgentStatus
from agents.review.reviewer import CodeReviewer, ReviewResult
from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)


class ReviewAgent(BaseAgent):
    """
    L5 — Code Review Agent

    Responsibilities:
      1. Receive testing handoff
      2. Load implementation files from disk
      3. Run LLM code review (scoring 0-100)
      4. Generate PR description
      5. Create GitHub PR (if token available)
      6. Auto-merge, regen-trigger, or human-gate based on score thresholds
      7. Hand off to Deploy Agent (L6)
    """

    agent_name = "review"
    layer = 5

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
    ):
        super().__init__(store=store, router=router)

    async def run(self, feature_id: UUID, **kwargs) -> AgentResult:
        t0 = time.monotonic()
        thresholds = self._workflow.get("thresholds", {})
        auto_merge_score = float(thresholds.get("code_review_score_auto_merge", 0.85)) * 100
        regen_score     = float(thresholds.get("code_review_score_regen", 0.70)) * 100

        # 1. Receive handoff from Testing Agent
        handoff = await self.store.receive_handoff(
            feature_id=feature_id,
            agent="review",
        )
        if not handoff:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error="No handoff found from testing agent",
            )

        title        = handoff.get("title", "Untitled")
        file_paths   = handoff.get("file_paths", [])
        test_summary = {
            "passed":       handoff.get("passed", 0),
            "failed":       handoff.get("failed", 0),
            "coverage_pct": handoff.get("coverage_pct", 0.0),
            "sast_findings": handoff.get("sast_findings", 0),
        }
        impl_id      = handoff.get("impl_id", "")
        suite_id     = handoff.get("suite_id", "")
        is_breaking  = handoff.get("is_breaking_change", False)
        complexity   = handoff.get("complexity", "Unknown")
        redmine_id   = handoff.get("redmine_id")

        # 2. Load implementation files
        impl_files = _load_files(file_paths)
        if not impl_files:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error=f"No readable files for review: {file_paths}",
            )

        async with self.store.locked(f"feature:{feature_id}:review") as acquired:
            if not acquired:
                return AgentResult(
                    status=AgentStatus.SKIPPED,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    error="Review step already being processed",
                )

            return await self._run_locked(
                feature_id=feature_id,
                title=title,
                impl_files=impl_files,
                file_paths=file_paths,
                test_summary=test_summary,
                impl_id=impl_id,
                suite_id=suite_id,
                is_breaking=is_breaking,
                complexity=complexity,
                redmine_id=redmine_id,
                auto_merge_score=auto_merge_score,
                regen_score=regen_score,
                t0=t0,
            )

    async def _run_locked(
        self,
        *,
        feature_id: UUID,
        title: str,
        impl_files: list[dict],
        file_paths: list[str],
        test_summary: dict,
        impl_id: str,
        suite_id: str,
        is_breaking: bool,
        complexity: str,
        redmine_id: int | None,
        auto_merge_score: float,
        regen_score: float,
        t0: float,
    ) -> AgentResult:

        past_context = await self._fetch_past_context(title)

        run_id = await self.begin(
            feature_id=feature_id,
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_summary=(
                f"title={title!r}  files={len(impl_files)}  "
                f"auto_merge>={auto_merge_score}  regen<{regen_score}"
            ),
        )

        try:
            reviewer = CodeReviewer(self._router)

            # 3. LLM code review
            review = await reviewer.review(
                feature_id=feature_id,
                title=title,
                impl_files=impl_files,
                test_summary=test_summary,
                existing_context=past_context,
            )

            # 4. Generate PR description
            pr_body = await reviewer.generate_pr_description(
                feature_id=feature_id,
                title=title,
                review=review,
                test_summary=test_summary,
                is_breaking=is_breaking,
                file_paths=file_paths,
            )

            # 5. Create GitHub PR (best-effort)
            branch = f"feature/{feature_id}"
            github_token = os.environ.get("GITHUB_TOKEN")
            repo_name = os.environ.get("GITHUB_REPO", "")
            pr = await reviewer.create_github_pr(
                repo_name=repo_name,
                title=f"feat: {title}",
                body=pr_body,
                branch=branch,
                github_token=github_token,
            )

            await self.store.advance_feature(feature_id, "review")

            # 6. Record decision
            outcome = _decide_outcome(review.score, auto_merge_score, regen_score)
            await self.store.record_decision(
                run_id=run_id,
                feature_id=feature_id,
                agent=self.agent_name,
                decision_type="code_review",
                summary=(
                    f"Review score: {review.score}/100  approved={review.approved}  "
                    f"outcome={outcome}  pr={pr.pr_number}"
                ),
                rationale=review.summary[:500],
                outcome=outcome,
            )

            # Update implementation node with PR number
            if pr.pr_number:
                await self.store.graph.update_node_status(
                    "Implementation", UUID(impl_id) if impl_id else uuid4(),
                    f"pr_open:{pr.pr_number}",
                )

            duration_ms = int((time.monotonic() - t0) * 1000)
            cost = review.router_response.cost_usd

            # 7. Hand off payload (used regardless of gate/success)
            handoff_payload = {
                "impl_id": impl_id,
                "suite_id": suite_id,
                "title": title,
                "file_paths": file_paths,
                "pr_number": pr.pr_number,
                "pr_url": pr.pr_url,
                "branch": branch,
                "review_score": review.score,
                "is_breaking_change": is_breaking,
                "complexity": complexity,
                "redmine_id": redmine_id,
            }
            await self.store.handoff(
                feature_id=feature_id,
                from_agent="review",
                to_agent="deploy",
                payload=handoff_payload,
            )

            # 8. Decide action
            if outcome == "regen":
                # Trigger regen: fail so pipeline can re-run implementation
                return await self.fail(
                    run_id=run_id,
                    feature_id=feature_id,
                    error=(
                        f"Review score {review.score:.0f} below regen threshold {regen_score:.0f}. "
                        f"Must-fix items: {'; '.join(review.must_fix[:3])}"
                    ),
                    cost_usd=cost,
                    duration_ms=duration_ms,
                )

            if outcome == "auto_merge":
                return await self.succeed(
                    run_id=run_id,
                    feature_id=feature_id,
                    output=handoff_payload,
                    cost_usd=cost,
                    duration_ms=duration_ms,
                    output_summary=f"Auto-approved: score={review.score:.0f} PR#{pr.pr_number}",
                )

            # Gate for human decision
            return await self.gate(
                run_id=run_id,
                feature_id=feature_id,
                gate_type="code_review_decision",
                message=(
                    f"Code review for '{title}': score={review.score:.0f}/100. "
                    f"{'PR #' + str(pr.pr_number) + ' ready.' if pr.pr_number else 'No PR created.'} "
                    f"Review and approve or reject."
                ),
                trigger_reason=f"score={review.score:.0f} between regen={regen_score:.0f} and auto_merge={auto_merge_score:.0f}",
                payload=handoff_payload,
                cost_usd=cost,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("ReviewAgent failed for feature %s", feature_id)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def _fetch_past_context(self, title: str) -> str | None:
        try:
            hits = await self.store.remember(title, limit=2, artifact_type="implementation")
            if not hits:
                return None
            return "\n\n".join(
                f"[Past review pattern, similarity={h['similarity']:.2f}]\n{h['content']}"
                for h in hits
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decide_outcome(score: float, auto_merge: float, regen: float) -> str:
    """Return "auto_merge", "gate", or "regen"."""
    if score >= auto_merge:
        return "auto_merge"
    if score < regen:
        return "regen"
    return "gate"


def _load_files(file_paths: list[str]) -> list[dict]:
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
    parser = argparse.ArgumentParser(description="AI SDLC — Review Agent (L5)")
    parser.add_argument("--feature-id", "-f", required=True, type=UUID)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with ReviewAgent() as agent:
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
