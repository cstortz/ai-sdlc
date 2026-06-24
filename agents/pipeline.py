"""
agents/pipeline.py — Sequential pipeline orchestrator.

Runs agents in order (L1 → L2 → … → L7), pausing at human gates and
resuming on approval. Publishes pipeline events to Redis throughout.

Usage:
    python -m agents.pipeline --prompt "Build a user login system"
    python -m agents.pipeline --redmine-id 42 --resume-feature <uuid>

Architecture note:
    Each agent publishes its completion to CHANNEL_PIPELINE (Redis).
    When swapping to event-driven in Phase 3+, replace the sequential
    calls below with subscribers on that channel — the agents themselves
    don't change.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from uuid import UUID

from agents.base import AgentResult, AgentStatus, CHANNEL_PIPELINE
from agents.intake.agent import IntakeAgent
from agents.architecture.agent import ArchitectureAgent
from agents.implementation.agent import ImplementationAgent
from agents.testing.agent import TestingAgent
from agents.review.agent import ReviewAgent
from agents.deploy.agent import DeployAgent
from agents.monitor.agent import MonitorAgent
from context_store import ContextStore

logger = logging.getLogger(__name__)

# Agents registered in pipeline order (layer 1 → N).
# Add Architecture, Implementation, etc. here as they are built.
PIPELINE_STAGES: list[tuple[str, type]] = [
    ("intake",        IntakeAgent),
    ("architecture",   ArchitectureAgent),
    ("implementation", ImplementationAgent),
    ("testing",        TestingAgent),
    ("review",         ReviewAgent),
    ("deploy",         DeployAgent),
    ("monitor",        MonitorAgent),
    # ("testing",      TestingAgent),
    # ("review",       ReviewAgent),
    # ("deploy",       DeployAgent),
    # ("monitor",      MonitorAgent),
]

# Gate polling config
GATE_POLL_INTERVAL  = 30   # seconds between gate checks
GATE_TIMEOUT_HOURS  = 48   # from workflow.yaml default


class Pipeline:
    """
    Sequential pipeline orchestrator.

    Runs each registered agent in order, sharing a single ContextStore
    and pausing at every human gate until approved.
    """

    def __init__(self, store: ContextStore | None = None):
        self._owns_store = store is None
        self._store = store

    async def __aenter__(self) -> "Pipeline":
        if self._store is None:
            self._store = ContextStore()
            await self._store.connect()
        return self

    async def __aexit__(self, *_) -> None:
        if self._owns_store and self._store:
            await self._store.close()

    @property
    def store(self) -> ContextStore:
        assert self._store, "Pipeline not connected"
        return self._store

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        prompt: str | None = None,
        redmine_id: int | None = None,
        feature_id: UUID | None = None,
        start_at: str = "intake",
        skip_interview: bool = False,
    ) -> dict[str, AgentResult]:
        """
        Execute the pipeline from `start_at` stage onward.

        Args:
            prompt:          NL feature description (intake trigger).
            redmine_id:      Redmine issue ID (intake trigger).
            feature_id:      Resume an existing feature from mid-pipeline.
            start_at:        Agent name to start from (default: "intake").
            skip_interview:  Pass through to IntakeAgent.

        Returns:
            Dict of {agent_name: AgentResult} for every stage that ran.
        """
        results: dict[str, AgentResult] = {}
        started = False

        for stage_name, AgentClass in PIPELINE_STAGES:
            if stage_name == start_at:
                started = True
            if not started:
                continue

            logger.info("Pipeline: starting %s", stage_name)
            agent = AgentClass(store=self.store)

            # Build stage-specific kwargs
            kwargs: dict = {}
            if stage_name == "intake":
                kwargs = dict(
                    prompt=prompt,
                    redmine_id=redmine_id,
                    feature_id=feature_id,
                    skip_interview=skip_interview,
                )
            else:
                kwargs = dict(feature_id=feature_id)

            result = await agent.run(**kwargs)
            results[stage_name] = result

            # Carry feature_id forward to subsequent stages
            if feature_id is None:
                feature_id = result.feature_id

            if result.status == AgentStatus.FAILED:
                logger.error("Pipeline halted: %s failed — %s", stage_name, result.error)
                break

            if result.status == AgentStatus.GATE_WAIT:
                approved = await self._handle_gate(result, stage_name)
                if not approved:
                    logger.warning("Pipeline halted: gate rejected/timed out at %s", stage_name)
                    break
                # Gate approved — continue to next stage

            if result.status == AgentStatus.SKIPPED:
                logger.warning("Pipeline: %s skipped — %s", stage_name, result.error)
                break

        return results

    async def _handle_gate(self, result: AgentResult, stage: str) -> bool:
        """
        Wait for a human gate to be resolved.

        In Phase 1 (CLI), this polls Postgres every GATE_POLL_INTERVAL seconds.
        The human approves by calling pipeline.approve_gate(gate_id) from
        a separate terminal or the Redmine interface.

        Returns True if approved, False if rejected or timed out.
        """
        gate_id = result.gate_id
        feature_id = result.feature_id
        logger.info(
            "Pipeline paused at '%s' gate (gate_id=%s). "
            "Approve with:\n  python -m agents.pipeline --approve-gate %s",
            stage, gate_id, gate_id,
        )

        print(f"\n{'═' * 60}")
        print(f"  ⏸  HUMAN GATE — {stage.upper()}")
        print(f"{'═' * 60}")
        print(f"  Gate ID:    {gate_id}")
        print(f"  Feature:    {feature_id}")
        print(f"\n  To approve:")
        print(f"    python -m agents.pipeline --approve-gate {gate_id}")
        print(f"\n  To reject:")
        print(f"    python -m agents.pipeline --reject-gate {gate_id}")
        print(f"\n  Waiting (checking every {GATE_POLL_INTERVAL}s, timeout {GATE_TIMEOUT_HOURS}h)…")
        print(f"{'═' * 60}\n")

        elapsed = 0.0
        timeout_s = GATE_TIMEOUT_HOURS * 3600

        while elapsed < timeout_s:
            await asyncio.sleep(GATE_POLL_INTERVAL)
            elapsed += GATE_POLL_INTERVAL

            # Check gate status in Postgres
            pending = await self.store.pending_gates(feature_id)
            still_pending = any(str(g["id"]) == str(gate_id) for g in pending)

            if not still_pending:
                # Gate was resolved — determine outcome
                outcome = await self._get_gate_outcome(gate_id)
                if outcome == "approved":
                    print(f"\n  ✓  Gate approved — continuing pipeline.\n")
                    return True
                else:
                    print(f"\n  ✗  Gate {outcome} — pipeline halted.\n")
                    return False

        logger.warning("Gate %s timed out after %dh", gate_id, GATE_TIMEOUT_HOURS)
        return False

    async def _get_gate_outcome(self, gate_id: UUID) -> str:
        """Query Postgres for the resolved gate status."""
        row = await self.store.pg.pool.fetchrow(
            "SELECT status FROM human_gates WHERE id = $1", gate_id
        )
        return row["status"] if row else "expired"

    # ------------------------------------------------------------------
    # Gate management (called from CLI)
    # ------------------------------------------------------------------

    async def approve_gate(self, gate_id: UUID, notes: str | None = None) -> None:
        await self.store.resolve_gate(gate_id, approved=True, notes=notes)
        logger.info("Gate %s approved", gate_id)
        print(f"  ✓  Gate {gate_id} approved.")

    async def reject_gate(self, gate_id: UUID, notes: str | None = None) -> None:
        await self.store.resolve_gate(gate_id, approved=False, notes=notes)
        logger.info("Gate %s rejected", gate_id)
        print(f"  ✗  Gate {gate_id} rejected.")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="AI SDLC — Pipeline Orchestrator")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt", "-p",        help="Start pipeline with a NL prompt")
    mode.add_argument("--redmine-id", "-r",    type=int, help="Start pipeline from a Redmine ticket")
    mode.add_argument("--resume",              type=UUID, metavar="FEATURE_ID", help="Resume pipeline from a feature UUID")
    mode.add_argument("--approve-gate",        type=UUID, metavar="GATE_ID",   help="Approve a pending human gate")
    mode.add_argument("--reject-gate",         type=UUID, metavar="GATE_ID",   help="Reject a pending human gate")
    parser.add_argument("--start-at",          default="intake", help="Stage name to start from (default: intake)")
    parser.add_argument("--skip-interview",    action="store_true")
    parser.add_argument("--notes",             default=None, help="Notes for gate approve/reject")
    parser.add_argument("--log-level",         default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with Pipeline() as pipeline:

        if args.approve_gate:
            await pipeline.approve_gate(args.approve_gate, notes=args.notes)
            return

        if args.reject_gate:
            await pipeline.reject_gate(args.reject_gate, notes=args.notes)
            return

        results = await pipeline.run(
            prompt=args.prompt,
            redmine_id=args.redmine_id,
            feature_id=args.resume,
            start_at=args.start_at,
            skip_interview=args.skip_interview,
        )

    print(f"\n{'═' * 60}  PIPELINE SUMMARY  {'═' * 60}")
    for stage, result in results.items():
        icon = {"success": "✓", "gate_wait": "⏸", "failed": "✗", "skipped": "⊘"}.get(result.status.value, "?")
        print(f"  {icon}  {stage:<20} {result.status.value:<12} cost=${result.cost_usd:.6f}")
    print(f"{'═' * 140}\n")


if __name__ == "__main__":
    asyncio.run(_main())
