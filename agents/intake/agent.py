"""
agents/intake/agent.py — L1 Intake Agent

Accepts a natural-language prompt or a Redmine ticket ID, interviews the
human to collect structured requirements, generates a PRD, stores all
artifacts in the context store, then creates a human approval gate before
the Architecture Agent (L2) can begin.

Run standalone:
    python -m agents.intake.agent --prompt "Build a user login system"
    python -m agents.intake.agent --redmine-id 42

Or use from pipeline.py:
    result = await IntakeAgent(store=store).run(feature_id=fid, prompt="...")
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path
from uuid import UUID

from agents.base import BaseAgent, AgentResult, AgentStatus
from agents.intake.interviewer import BaseInterviewer, InterviewMode, create_interviewer
from agents.intake.prd_writer import InterviewAnswers, PRDWriter
from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)


class IntakeAgent(BaseAgent):
    """
    L1 — Intake Agent

    Responsibilities:
      1. Accept NL prompt or Redmine ticket
      2. Derive initial title from the input
      3. Run the interview (CLI or Redmine)
      4. Generate PRD via LLM
      5. Register feature + PRD node in context store
      6. Store PRD embeddings for semantic search
      7. Hand off context to Architecture Agent
      8. Create human approval gate (always in Phase 1)
    """

    agent_name = "intake"
    layer = 1

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
        interview_mode: InterviewMode | str = InterviewMode.CLI,
        interviewer: BaseInterviewer | None = None,
    ):
        super().__init__(store=store, router=router)
        self._interview_mode = InterviewMode(interview_mode)
        self._interviewer = interviewer  # injected in tests; created lazily otherwise

    @property
    def interviewer(self) -> BaseInterviewer:
        if self._interviewer is None:
            self._interviewer = create_interviewer(self._interview_mode)
        return self._interviewer

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        feature_id: UUID | None = None,
        *,
        prompt: str | None = None,
        redmine_id: int | None = None,
        skip_interview: bool = False,
    ) -> AgentResult:
        """
        Run the intake pipeline.

        Args:
            feature_id:      Pre-assigned UUID (pipeline sets this); created if None.
            prompt:          Free-text feature description.
            redmine_id:      Redmine issue ID (alternative to prompt).
            skip_interview:  Skip clarifying questions (useful in tests or when
                             the prompt is already fully specified).

        Returns:
            AgentResult with status SUCCESS (gate created) or FAILED.
        """
        t0 = time.monotonic()
        stage_cfg = self._stage_config("intake")
        max_questions = stage_cfg.get("max_clarifying_questions", 3)

        if not prompt and not redmine_id:
            raise ValueError("IntakeAgent.run() requires either prompt= or redmine_id=")

        # 1. Resolve/create feature
        if feature_id is None:
            raw_title = self._derive_title(prompt or f"Redmine #{redmine_id}")
            feature_id = await self.store.register_feature(
                title=raw_title,
                redmine_id=redmine_id,
                description=prompt,
            )
            logger.info("Registered new feature: %s  title=%r", feature_id, raw_title)
        else:
            feature_rec = await self.store.pg.get_feature(feature_id)
            raw_title = feature_rec["title"] if feature_rec else "Untitled"

        # Acquire distributed lock — prevent duplicate runs on same feature
        async with self.store.locked(f"feature:{feature_id}") as acquired:
            if not acquired:
                logger.warning("intake: feature %s already being processed", feature_id)
                return AgentResult(
                    status=AgentStatus.SKIPPED,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    error="Feature already being processed by another run",
                )

            return await self._run_locked(
                feature_id=feature_id,
                raw_title=raw_title,
                prompt=prompt or f"Redmine ticket #{redmine_id}",
                redmine_id=redmine_id,
                max_questions=max_questions,
                skip_interview=skip_interview,
                t0=t0,
            )

    async def _run_locked(
        self,
        *,
        feature_id: UUID,
        raw_title: str,
        prompt: str,
        redmine_id: int | None,
        max_questions: int,
        skip_interview: bool,
        t0: float,
    ) -> AgentResult:
        # 2. Retrieve any past context for this feature (re-runs / revisions)
        past_context = await self._fetch_past_context(feature_id)

        # 3. Interview
        answers = await self._interview(
            raw_title=raw_title,
            prompt=prompt,
            max_questions=max_questions,
            skip=skip_interview,
        )

        # 4. Log run start (we now know the model we'll use)
        run_id = await self.begin(
            feature_id=feature_id,
            model="claude-sonnet-4-6",     # intake profile primary; actual model logged by router
            provider="anthropic",
            input_summary=f"title={answers.title!r}  prompt={prompt[:100]!r}",
        )

        try:
            # 5. Generate PRD
            await self.interviewer.show("✦ Generating PRD — this may take a moment…")
            writer = PRDWriter(self._router)
            prd = await writer.write(
                feature_id=feature_id,
                answers=answers,
                existing_context=past_context,
            )

            # 6. Update feature title to the cleaned PRD title
            await self.store.advance_feature(feature_id, "intake")

            # 7. Register PRD node in graph
            prd_id = _new_uuid()
            await self.store.graph.create_prd(
                id=prd_id,
                feature_id=feature_id,
                file_path=str(prd.file_path.relative_to(Path.cwd()) if prd.file_path.is_relative_to(Path.cwd()) else prd.file_path),
                version=1,
                status="draft",
            )

            # 8. Embed PRD for semantic search
            await self.store.memorize(
                prd.content,
                feature_id=feature_id,
                artifact_type="prd",
                artifact_id=str(prd.file_path),
            )

            # 9. Show PRD to human
            await self.interviewer.show(
                f"\n{'═' * 60}\n"
                f"  PRD DRAFT\n"
                f"{'═' * 60}\n"
                f"{prd.content}\n"
                f"{'═' * 60}\n"
                f"  Saved to: {prd.file_path}\n"
                f"{'═' * 60}"
            )

            # 10. Record decision
            await self.store.record_decision(
                run_id=run_id,
                feature_id=feature_id,
                agent=self.agent_name,
                decision_type="prd_generated",
                summary=f"PRD generated for '{answers.title}'",
                rationale=f"Interview answers: {answers.to_prompt()[:500]}",
                outcome="pending_human_approval",
            )

            # 11. Hand off context to Architecture Agent
            await self.store.handoff(
                feature_id=feature_id,
                from_agent="intake",
                to_agent="architecture",
                payload={
                    "prd_file": str(prd.file_path),
                    "prd_id": str(prd_id),
                    "title": answers.title,
                    "redmine_id": redmine_id,
                },
            )

            # 12. Create human approval gate (always in Phase 1)
            duration_ms = int((time.monotonic() - t0) * 1000)
            cost = prd.router_response.cost_usd

            return await self.gate(
                run_id=run_id,
                feature_id=feature_id,
                gate_type="human_approval",
                message=f"Review and approve PRD for '{answers.title}' before architecture begins.",
                trigger_reason="always — intake always requires human approval in Phase 1",
                payload={
                    "prd_file": str(prd.file_path),
                    "prd_id": str(prd_id),
                    "title": answers.title,
                },
                cost_usd=cost,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("IntakeAgent failed for feature %s", feature_id)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    # ------------------------------------------------------------------
    # Interview
    # ------------------------------------------------------------------

    async def _interview(
        self,
        *,
        raw_title: str,
        prompt: str,
        max_questions: int,
        skip: bool,
    ) -> InterviewAnswers:
        """Conduct the clarifying-question interview and return structured answers."""

        answers = InterviewAnswers(title=raw_title, raw_prompt=prompt)

        if skip:
            return answers

        await self.interviewer.show(
            f"\n{'═' * 60}\n"
            f"  AI SDLC — Intake Agent\n"
            f"{'═' * 60}\n"
            f"  Feature: {raw_title}\n"
            f"  I'll ask up to {max_questions} clarifying questions.\n"
            f"  Press Enter to skip any question.\n"
            f"{'═' * 60}"
        )

        questions = [
            ("problem",            "What specific problem does this feature solve? Who has this problem?"),
            ("users",              "Who are the primary users of this feature? Describe their role and context."),
            ("success_criteria",   "What does success look like? How will you measure it?"),
            ("out_of_scope",       "What should this feature explicitly NOT do? Any known constraints?"),
            ("additional_context", "Any other context, deadlines, dependencies, or references I should know?"),
        ]

        asked = 0
        for field_name, question in questions:
            if asked >= max_questions:
                break
            answer = await self.interviewer.ask(question)
            if answer:
                setattr(answers, field_name, answer)
                asked += 1

        # Allow title refinement
        refined_title = await self.interviewer.ask(
            f"Confirm or refine the feature title (current: '{raw_title}'):",
        )
        if refined_title:
            answers.title = refined_title

        return answers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_past_context(self, feature_id: UUID) -> str | None:
        """Semantic search for past decisions relevant to this feature."""
        try:
            feature = await self.store.pg.get_feature(feature_id)
            if not feature:
                return None
            hits = await self.store.remember(
                feature["title"],
                limit=3,
                artifact_type="prd",
            )
            if not hits:
                return None
            return "\n\n".join(
                f"[Past PRD excerpt, similarity={h['similarity']:.2f}]\n{h['content']}"
                for h in hits
            )
        except Exception:
            return None

    @staticmethod
    def _derive_title(prompt: str) -> str:
        """Extract a short title from the raw prompt (first sentence, max 80 chars)."""
        first_sentence = prompt.split(".")[0].split("\n")[0].strip()
        return first_sentence[:80] if first_sentence else prompt[:80]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _new_uuid() -> UUID:
    from uuid import uuid4
    return uuid4()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="AI SDLC — Intake Agent (L1)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", "-p", help="Free-text feature description")
    group.add_argument("--redmine-id", "-r", type=int, help="Redmine issue ID")
    parser.add_argument("--skip-interview", action="store_true", help="Skip clarifying questions")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with IntakeAgent() as agent:
        result = await agent.run(
            prompt=args.prompt,
            redmine_id=args.redmine_id,
            skip_interview=args.skip_interview,
        )

    print(f"\n{'═' * 60}")
    print(f"  Status:     {result.status.value}")
    print(f"  Feature ID: {result.feature_id}")
    if result.gate_id:
        print(f"  Gate ID:    {result.gate_id}")
        print(f"  Next step:  Approve the PRD, then run the Architecture Agent.")
    elif result.error:
        print(f"  Error:      {result.error}")
    print(f"  Cost:       ${result.cost_usd:.6f}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(_main())
