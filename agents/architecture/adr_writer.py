"""
agents/architecture/adr_writer.py — ADR generation from PRD content.

Takes the PRD and any relevant past decisions, calls the LLM (architecture
profile) to produce a structured Architecture Decision Record (ADR), then
writes it to docs/adrs/.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from router import ModelRouter, RouterResponse

logger = logging.getLogger(__name__)

ADR_DIR = Path(__file__).parent.parent.parent / "docs" / "adrs"

# ---------------------------------------------------------------------------
# ADR system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a principal software architect in an AI-powered SDLC pipeline.
Your job is to produce a clear, opinionated Architecture Decision Record (ADR)
based on the provided PRD and context.

Output ONLY the ADR markdown document — no preamble, no explanation.

The document must follow this exact structure:

# ADR: {title}

**Status:** Proposed
**Date:** {date}
**Feature:** {feature_id}

## Context
What is the background and forces at play that motivated this decision?
Include relevant constraints, existing systems, and non-functional requirements.

## Decision
State the architectural decision clearly. What approach was chosen and why?
Be specific — name technologies, patterns, and integration points.

## Consequences
### Positive
- What becomes easier or better because of this decision?

### Negative / Trade-offs
- What becomes harder or what do we accept as a trade-off?

### Risks
- What could go wrong? What should be monitored?

## Alternatives Considered
For each rejected alternative, briefly explain why it was not chosen.

## Implementation Notes
Key technical details the engineering team needs to implement this decision:
- Data models / schema changes (if any)
- API surface changes (if any)
- Infrastructure requirements (if any)
- Security considerations (always include this section)
- Estimated complexity: [Low | Medium | High]

## Breaking Change Assessment
Is this a breaking change? Answer YES or NO, then explain.
A breaking change is any change that modifies existing API contracts, database
schemas, authentication flows, or deployed service interfaces.
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ADRResult:
    """Output of the ADR writer."""
    feature_id: UUID
    title: str
    content: str           # Full markdown text
    file_path: Path
    is_breaking_change: bool
    complexity: str        # "Low" | "Medium" | "High"
    router_response: RouterResponse


# ---------------------------------------------------------------------------
# ADR writer
# ---------------------------------------------------------------------------


class ADRWriter:
    """
    Generates an ADR from a PRD via the LLM (architecture profile).

    Usage:
        writer = ADRWriter(router)
        result = await writer.write(feature_id=fid, title=title, prd_content=prd)
    """

    def __init__(self, router: ModelRouter):
        self._router = router

    async def write(
        self,
        feature_id: UUID,
        title: str,
        prd_content: str,
        existing_context: str | None = None,
        version: int = 1,
    ) -> ADRResult:
        """
        Generate the ADR and write it to docs/adrs/{feature_id}.md.

        Args:
            feature_id:       UUID of the feature.
            title:            Feature/ADR title.
            prd_content:      Full PRD markdown text.
            existing_context: Related past ADRs/decisions from semantic search.
            version:          ADR version number (for revisions).

        Returns:
            ADRResult with the content, file path, and breaking-change flag.
        """
        from datetime import date
        messages = self._build_messages(title, feature_id, prd_content, existing_context)

        logger.info("Generating ADR for feature %s: %r", feature_id, title)
        response = await self._router.complete(
            profile="architecture",
            messages=messages,
            system=SYSTEM_PROMPT,
        )

        content = _clean_content(response.content)
        is_breaking = _detect_breaking_change(content)
        complexity = _detect_complexity(content)
        file_path = _write_file(feature_id, content, version)

        logger.info(
            "ADR written: %s  breaking=%s  complexity=%s  cost=$%.6f",
            file_path, is_breaking, complexity, response.cost_usd,
        )
        return ADRResult(
            feature_id=feature_id,
            title=title,
            content=content,
            file_path=file_path,
            is_breaking_change=is_breaking,
            complexity=complexity,
            router_response=response,
        )

    async def revise(
        self,
        feature_id: UUID,
        title: str,
        existing_adr: str,
        feedback: str,
        version: int = 2,
    ) -> ADRResult:
        """
        Revise an existing ADR based on human feedback.
        Called when the gate reviewer rejects the architecture proposal.
        """
        revision_prompt = (
            f"Here is the current ADR:\n\n{existing_adr}\n\n"
            f"The reviewer provided this feedback:\n{feedback}\n\n"
            "Revise the ADR to address this feedback. Preserve all sections; "
            "only update what the feedback requests. Update the Status to 'Revised'."
        )
        response = await self._router.complete(
            profile="architecture",
            messages=[{"role": "user", "content": revision_prompt}],
            system=SYSTEM_PROMPT,
        )
        content = _clean_content(response.content)
        is_breaking = _detect_breaking_change(content)
        complexity = _detect_complexity(content)
        file_path = _write_file(feature_id, content, version)

        return ADRResult(
            feature_id=feature_id,
            title=title,
            content=content,
            file_path=file_path,
            is_breaking_change=is_breaking,
            complexity=complexity,
            router_response=response,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        title: str,
        feature_id: UUID,
        prd_content: str,
        existing_context: str | None,
    ) -> list[dict]:
        user_content = (
            f"Feature title: {title}\n"
            f"Feature ID: {feature_id}\n\n"
            f"PRD:\n{prd_content}"
        )
        if existing_context:
            user_content = (
                f"Relevant past ADRs and decisions:\n{existing_context}\n\n"
                f"---\n\n{user_content}"
            )
        return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Module-level helpers (testable as pure functions)
# ---------------------------------------------------------------------------


def _clean_content(raw: str) -> str:
    """Strip any accidental wrapper code fences."""
    cleaned = re.sub(r"^```(?:markdown)?\n?", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    return cleaned.strip()


def _detect_breaking_change(content: str) -> bool:
    """
    Parse the 'Breaking Change Assessment' section.
    Returns True if the LLM wrote YES (case-insensitive).
    """
    section_match = re.search(
        r"##\s+Breaking Change Assessment\s*\n(.+?)(?=\n##|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        return False
    section_text = section_match.group(1).strip()
    return bool(re.match(r"yes\b", section_text, re.IGNORECASE))


def _detect_complexity(content: str) -> str:
    """
    Parse 'Estimated complexity: [Low|Medium|High]' from Implementation Notes.
    Returns "Unknown" if not found.
    """
    match = re.search(
        r"[Ee]stimated\s+complexity[:\s]+\*{0,2}(Low|Medium|High)\*{0,2}",
        content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).capitalize()
    return "Unknown"


def _write_file(feature_id: UUID, content: str, version: int = 1) -> Path:
    ADR_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_v{version}" if version > 1 else ""
    path = ADR_DIR / f"{feature_id}{suffix}.md"
    path.write_text(content, encoding="utf-8")
    return path
