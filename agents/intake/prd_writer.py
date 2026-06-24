"""
agents/intake/prd_writer.py — PRD generation from structured interview answers.

Takes the collected interview answers and calls the LLM (intake profile)
to produce a structured PRD in Markdown, then writes it to docs/prds/.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from router import ModelRouter, RouterResponse

logger = logging.getLogger(__name__)

PRD_DIR = Path(__file__).parent.parent.parent / "docs" / "prds"

# ---------------------------------------------------------------------------
# PRD system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Requirements Engineer in an AI-powered SDLC pipeline.
Your job is to transform raw interview answers into a clean, structured PRD.

Output ONLY the PRD markdown document — no preamble, no explanation.

The document must follow this exact structure:

# PRD: {title}

## Problem Statement
One to three sentences. What problem does this solve, and why does it matter?

## Goals
Numbered list of concrete, measurable goals. Each goal must be achievable and testable.

## User Stories
For each story use this format exactly:
**As a** <role>, **I want** <action> **so that** <benefit>.

*Acceptance Criteria:*
- Given <context>, when <action>, then <outcome>.
(2–4 criteria per story)

## Out of Scope
What this feature explicitly does NOT include. Be specific.

## Open Questions
Any unresolved questions that must be answered before implementation begins.
If none, write "None."

## Notes
Any additional context, constraints, or references the engineering team should know.
If none, omit this section.
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InterviewAnswers:
    """Structured answers collected by the interviewer."""
    title: str
    raw_prompt: str
    problem: str = ""
    users: str = ""
    success_criteria: str = ""
    out_of_scope: str = ""
    additional_context: str = ""

    def to_prompt(self) -> str:
        """Format answers into a user message for the LLM."""
        parts = [f"Feature title: {self.title}", f"Original request: {self.raw_prompt}"]
        if self.problem:
            parts.append(f"Problem being solved: {self.problem}")
        if self.users:
            parts.append(f"Target users: {self.users}")
        if self.success_criteria:
            parts.append(f"Success looks like: {self.success_criteria}")
        if self.out_of_scope:
            parts.append(f"Out of scope: {self.out_of_scope}")
        if self.additional_context:
            parts.append(f"Additional context: {self.additional_context}")
        return "\n\n".join(parts)


@dataclass
class PRDResult:
    """Output of the PRD writer."""
    feature_id: UUID
    title: str
    content: str          # Full markdown text
    file_path: Path
    router_response: RouterResponse


# ---------------------------------------------------------------------------
# PRD writer
# ---------------------------------------------------------------------------

class PRDWriter:
    """
    Generates a PRD from interview answers using the LLM.

    Usage:
        writer = PRDWriter(router)
        result = await writer.write(feature_id=fid, answers=answers)
    """

    def __init__(self, router: ModelRouter):
        self._router = router

    async def write(
        self,
        feature_id: UUID,
        answers: InterviewAnswers,
        existing_context: str | None = None,
    ) -> PRDResult:
        """
        Generate the PRD markdown and write it to docs/prds/{feature_id}.md.

        Args:
            feature_id:       UUID of the feature (used as filename).
            answers:          Structured interview answers.
            existing_context: Optional relevant past PRDs/decisions for context.

        Returns:
            PRDResult with the content and file path.
        """
        messages = self._build_messages(answers, existing_context)

        logger.info("Generating PRD for feature %s: %r", feature_id, answers.title)
        response = await self._router.complete(
            profile="intake",
            messages=messages,
            system=SYSTEM_PROMPT,
        )

        content = self._clean_content(response.content)
        file_path = self._write_file(feature_id, content)

        logger.info(
            "PRD written: %s  tokens=%d+%d  cost=$%.6f",
            file_path, response.input_tokens, response.output_tokens, response.cost_usd,
        )
        return PRDResult(
            feature_id=feature_id,
            title=answers.title,
            content=content,
            file_path=file_path,
            router_response=response,
        )

    async def revise(
        self,
        feature_id: UUID,
        answers: InterviewAnswers,
        existing_prd: str,
        feedback: str,
    ) -> PRDResult:
        """
        Revise an existing PRD based on human feedback.
        Called when the human rejects the initial PRD at the approval gate.
        """
        revision_prompt = (
            f"Here is the current PRD:\n\n{existing_prd}\n\n"
            f"The reviewer provided this feedback:\n{feedback}\n\n"
            "Please revise the PRD to address this feedback. "
            "Keep all sections that are correct; only update what the feedback requests."
        )
        messages = [{"role": "user", "content": revision_prompt}]

        response = await self._router.complete(
            profile="intake",
            messages=messages,
            system=SYSTEM_PROMPT,
        )
        content = self._clean_content(response.content)
        file_path = self._write_file(feature_id, content)

        return PRDResult(
            feature_id=feature_id,
            title=answers.title,
            content=content,
            file_path=file_path,
            router_response=response,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(answers: InterviewAnswers, existing_context: str | None) -> list[dict]:
        user_content = answers.to_prompt()
        if existing_context:
            user_content = (
                f"Relevant context from past decisions:\n{existing_context}\n\n"
                f"---\n\n{user_content}"
            )
        return [{"role": "user", "content": user_content}]

    @staticmethod
    def _clean_content(raw: str) -> str:
        """Strip any accidental wrapper text the LLM might add."""
        # Remove markdown code fences if the LLM wrapped the output
        cleaned = re.sub(r"^```(?:markdown)?\n?", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
        return cleaned.strip()

    @staticmethod
    def _write_file(feature_id: UUID, content: str) -> Path:
        PRD_DIR.mkdir(parents=True, exist_ok=True)
        path = PRD_DIR / f"{feature_id}.md"
        path.write_text(content, encoding="utf-8")
        return path
