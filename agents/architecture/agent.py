"""
agents/architecture/agent.py — L2 Architecture Agent

Reads the PRD handoff from the Intake Agent, retrieves relevant past ADRs
and decisions from semantic search, generates a new ADR via LLM, stores
all artifacts, then:
  - If breaking change OR complexity=High → creates human gate (always in Phase 1)
  - Otherwise → auto-approves if below threshold (future Phase 2+ behaviour;
    in Phase 1 the workflow config forces always_escalate=true)

Run standalone:
    python -m agents.architecture.agent --feature-id <uuid>

Or from pipeline.py via PIPELINE_STAGES.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from uuid import UUID, uuid4

from agents.base import BaseAgent, AgentResult, AgentStatus
from agents.architecture.adr_writer import ADRWriter, ADRResult
from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)


class ArchitectureAgent(BaseAgent):
    """
    L2 — Architecture Agent

    Responsibilities:
      1. Receive PRD handoff from Intake Agent
      2. Semantic search for related past ADRs/decisions
      3. Generate ADR via LLM (architecture profile)
      4. Register ADR node in traceability graph
      5. Embed ADR for future semantic search
      6. Hand off to Implementation Agent
      7. Create human gate (breaking changes, High complexity, or always in Phase 1)
    """

    agent_name = "architecture"
    layer = 2

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
    ):
        super().__init__(store=store, router=router)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        feature_id: UUID,
        **kwargs,
    ) -> AgentResult:
        """
        Run the architecture pipeline for an existing feature.

        Expects the Intake Agent's handoff to be waiting in context_snapshots.

        Args:
            feature_id: UUID of the feature to architect.

        Returns:
            AgentResult with status GATE_WAIT (needs approval) or FAILED.
        """
        t0 = time.monotonic()
        stage_cfg = self._stage_config("architecture")
        always_escalate = stage_cfg.get("breaking_change_always_escalate", True)

        # 1. Receive handoff from Intake Agent
        handoff = await self.store.receive_handoff(
            feature_id=feature_id,
            agent="architecture",
        )
        if not handoff:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error="No handoff found from intake agent — has intake completed?",
            )

        prd_file = handoff.get("prd_file", "")
        title = handoff.get("title", "Untitled")
        redmine_id = handoff.get("redmine_id")

        # 2. Load PRD content
        prd_content = _read_prd(prd_file)
        if not prd_content:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error=f"Could not read PRD file: {prd_file}",
            )

        # Acquire distributed lock
        async with self.store.locked(f"feature:{feature_id}:architecture") as acquired:
            if not acquired:
                logger.warning("architecture: feature %s already being processed", feature_id)
                return AgentResult(
                    status=AgentStatus.SKIPPED,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    error="Architecture step already being processed by another run",
                )

            return await self._run_locked(
                feature_id=feature_id,
                title=title,
                prd_content=prd_content,
                redmine_id=redmine_id,
                always_escalate=always_escalate,
                t0=t0,
            )

    async def _run_locked(
        self,
        *,
        feature_id: UUID,
        title: str,
        prd_content: str,
        redmine_id: int | None,
        always_escalate: bool,
        t0: float,
    ) -> AgentResult:

        # 3. Semantic search for past ADRs
        past_context = await self._fetch_past_context(title, prd_content)

        # 4. Log run start
        run_id = await self.begin(
            feature_id=feature_id,
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_summary=f"title={title!r}  prd_chars={len(prd_content)}",
        )

        try:
            # 5. Generate ADR
            writer = ADRWriter(self._router)
            adr = await writer.write(
                feature_id=feature_id,
                title=title,
                prd_content=prd_content,
                existing_context=past_context,
            )

            # 6. Advance feature status
            await self.store.advance_feature(feature_id, "architecture")

            # 7. Register ADR node in graph
            adr_id = uuid4()
            await self.store.graph.create_adr(
                id=adr_id,
                feature_id=feature_id,
                file_path=str(adr.file_path),
                decision_type="breaking_change" if adr.is_breaking_change else "standard",
                status="proposed",
            )

            # 8. Embed ADR for semantic search
            await self.store.memorize(
                adr.content,
                feature_id=feature_id,
                artifact_type="adr",
                artifact_id=str(adr.file_path),
            )

            # 9. Record decision
            trigger = _gate_trigger(adr, always_escalate)
            await self.store.record_decision(
                run_id=run_id,
                feature_id=feature_id,
                agent=self.agent_name,
                decision_type="adr_generated",
                summary=f"ADR generated for '{title}'  breaking={adr.is_breaking_change}  complexity={adr.complexity}",
                rationale=f"breaking_change={adr.is_breaking_change}, complexity={adr.complexity}, always_escalate={always_escalate}",
                outcome="pending_human_approval" if trigger else "auto_approved",
            )

            # 10. Hand off to Implementation Agent
            await self.store.handoff(
                feature_id=feature_id,
                from_agent="architecture",
                to_agent="implementation",
                payload={
                    "adr_file": str(adr.file_path),
                    "adr_id": str(adr_id),
                    "title": title,
                    "is_breaking_change": adr.is_breaking_change,
                    "complexity": adr.complexity,
                    "redmine_id": redmine_id,
                },
            )

            duration_ms = int((time.monotonic() - t0) * 1000)
            cost = adr.router_response.cost_usd

            # 11. Gate or auto-approve
            if trigger:
                return await self.gate(
                    run_id=run_id,
                    feature_id=feature_id,
                    gate_type="architecture_review",
                    message=_gate_message(adr, title, always_escalate),
                    trigger_reason=trigger,
                    payload={
                        "adr_file": str(adr.file_path),
                        "adr_id": str(adr_id),
                        "is_breaking_change": adr.is_breaking_change,
                        "complexity": adr.complexity,
                    },
                    cost_usd=cost,
                    duration_ms=duration_ms,
                )
            else:
                # Future Phase 2+: auto-approve low-risk architecture
                return await self.succeed(
                    run_id=run_id,
                    feature_id=feature_id,
                    output={
                        "adr_file": str(adr.file_path),
                        "adr_id": str(adr_id),
                        "is_breaking_change": adr.is_breaking_change,
                        "complexity": adr.complexity,
                    },
                    cost_usd=cost,
                    duration_ms=duration_ms,
                    output_summary=f"ADR auto-approved: '{title}'",
                )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("ArchitectureAgent failed for feature %s", feature_id)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_past_context(self, title: str, prd_content: str) -> str | None:
        """Semantic search for related past ADRs and decisions."""
        try:
            query = f"{title}\n\n{prd_content[:500]}"
            hits = await self.store.remember(query, limit=3, artifact_type="adr")
            if not hits:
                hits = await self.store.remember(query, limit=3)
            if not hits:
                return None
            return "\n\n".join(
                f"[Past ADR excerpt, similarity={h['similarity']:.2f}]\n{h['content']}"
                for h in hits
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _read_prd(prd_file: str) -> str | None:
    """Read PRD file content; return None on error."""
    try:
        path = Path(prd_file)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read PRD file %s: %s", prd_file, exc)
        return None


def _gate_trigger(adr: ADRResult, always_escalate: bool) -> str | None:
    """
    Returns a reason string if a human gate should be created, else None.

    Gate conditions (Phase 1: always_escalate=True means always gate):
      - always_escalate is True (workflow.yaml default in Phase 1)
      - is_breaking_change is True
      - complexity == "High"
    """
    if always_escalate:
        return "Phase 1: architecture always requires human approval"
    if adr.is_breaking_change:
        return "breaking_change=True requires human approval"
    if adr.complexity == "High":
        return "High complexity requires human approval"
    return None


def _gate_message(adr: ADRResult, title: str, always_escalate: bool) -> str:
    reasons = []
    if always_escalate:
        reasons.append("Phase 1 requires human review for all ADRs")
    if adr.is_breaking_change:
        reasons.append("breaking change detected")
    if adr.complexity == "High":
        reasons.append("high complexity")
    reason_str = "; ".join(reasons) if reasons else "standard review"
    return (
        f"Review and approve ADR for '{title}' before implementation begins. "
        f"Reason: {reason_str}. "
        f"Breaking change: {adr.is_breaking_change}. Complexity: {adr.complexity}."
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="AI SDLC — Architecture Agent (L2)")
    parser.add_argument("--feature-id", "-f", required=True, type=UUID, help="Feature UUID")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with ArchitectureAgent() as agent:
        result = await agent.run(feature_id=args.feature_id)

    print(f"\n{'═' * 60}")
    print(f"  Status:     {result.status.value}")
    print(f"  Feature ID: {result.feature_id}")
    if result.gate_id:
        print(f"  Gate ID:    {result.gate_id}")
        print(f"  Next step:  Approve the ADR, then run the Implementation Agent.")
    elif result.error:
        print(f"  Error:      {result.error}")
    print(f"  Cost:       ${result.cost_usd:.6f}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(_main())
