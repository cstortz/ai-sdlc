"""
agents/implementation/code_writer.py — Code generation from PRD + ADR.

Takes the PRD, ADR, and any relevant past implementations from semantic
search, calls the LLM (implementation profile) to produce a structured
implementation plan with actual code files, then writes them to src/.

The LLM is asked to produce a JSON manifest describing the files to create,
followed by the file contents. This module parses that manifest and writes
each file to disk.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from router import ModelRouter, RouterResponse

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent.parent.parent / "src"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior software engineer in an AI-powered SDLC pipeline.
Your job is to implement a feature based on the provided PRD and ADR.

Output a JSON object ONLY — no preamble, no explanation, no code fences.

The JSON must follow this exact schema:
{
  "summary": "One-sentence description of what was implemented.",
  "files": [
    {
      "path": "src/relative/path/to/file.py",
      "description": "What this file does.",
      "content": "Full file content as a string."
    }
  ],
  "notes": "Optional implementation notes, caveats, or follow-up items.",
  "estimated_test_coverage": "Brief statement on what unit tests should cover."
}

Rules:
- All file paths must be relative to the project root and start with "src/".
- Write complete, production-quality code — no placeholders, no TODO stubs.
- Include docstrings, type hints, and logging.
- Each file must be independently coherent.
- Follow the architectural decisions in the ADR exactly.
- Security considerations from the ADR MUST be implemented, not deferred.
- If the feature requires DB migrations, include them as src/migrations/*.sql.
"""

REVISION_SYSTEM_PROMPT = """\
You are a senior software engineer reviewing and fixing your own implementation.
The code reviewer found issues. Fix them and return the complete updated implementation.

Output the same JSON schema as before — no preamble, no explanation, no code fences.
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GeneratedFile:
    path: str           # relative to project root, e.g. "src/auth/jwt.py"
    description: str
    content: str
    abs_path: Path = field(init=False)

    def __post_init__(self):
        self.abs_path = SRC_DIR.parent / self.path


@dataclass
class ImplementationResult:
    """Output of the code writer."""
    feature_id: UUID
    title: str
    summary: str
    files: list[GeneratedFile]
    notes: str
    estimated_test_coverage: str
    router_response: RouterResponse
    iteration: int = 1


# ---------------------------------------------------------------------------
# Code writer
# ---------------------------------------------------------------------------


class CodeWriter:
    """
    Generates implementation files from PRD + ADR via LLM.

    Usage:
        writer = CodeWriter(router)
        result = await writer.write(feature_id=fid, title=title, prd=prd, adr=adr)
    """

    def __init__(self, router: ModelRouter):
        self._router = router

    async def write(
        self,
        feature_id: UUID,
        title: str,
        prd_content: str,
        adr_content: str,
        existing_context: str | None = None,
        iteration: int = 1,
    ) -> ImplementationResult:
        """
        Generate implementation files and write them to disk.

        Args:
            feature_id:       UUID of the feature.
            title:            Feature title.
            prd_content:      Full PRD markdown.
            adr_content:      Full ADR markdown.
            existing_context: Related past implementations from semantic search.
            iteration:        Which generation attempt this is (for logging).

        Returns:
            ImplementationResult with generated files.
        """
        messages = _build_messages(title, prd_content, adr_content, existing_context)

        logger.info(
            "Generating implementation for feature %s (iteration %d): %r",
            feature_id, iteration, title,
        )
        response = await self._router.complete(
            profile="implementation",
            messages=messages,
            system=SYSTEM_PROMPT,
        )

        result = _parse_response(response, feature_id, title, iteration)
        _write_files(result.files)

        logger.info(
            "Implementation written: %d file(s)  cost=$%.6f",
            len(result.files), response.cost_usd,
        )
        return result

    async def revise(
        self,
        feature_id: UUID,
        title: str,
        existing_result: ImplementationResult,
        review_feedback: str,
    ) -> ImplementationResult:
        """
        Revise implementation based on code review feedback.
        Called when review score is below the regen threshold.
        """
        files_summary = "\n".join(
            f"  {f.path}: {f.description}" for f in existing_result.files
        )
        revision_prompt = (
            f"Feature: {title}\n\n"
            f"Current implementation files:\n{files_summary}\n\n"
            f"Code review feedback:\n{review_feedback}\n\n"
            "Fix all issues identified in the review. "
            "Return the complete updated implementation for ALL files, "
            "not just the changed ones."
        )

        # Include existing file contents for context
        file_contents = "\n\n".join(
            f"=== {f.path} ===\n{f.content}" for f in existing_result.files
        )
        messages = [
            {"role": "user", "content": revision_prompt},
            {"role": "assistant", "content": "I'll review the feedback and fix all issues."},
            {"role": "user", "content": f"Here are the current file contents:\n\n{file_contents}"},
        ]

        response = await self._router.complete(
            profile="implementation",
            messages=messages,
            system=REVISION_SYSTEM_PROMPT,
        )

        result = _parse_response(
            response, feature_id, title, existing_result.iteration + 1
        )
        _write_files(result.files)
        return result


# ---------------------------------------------------------------------------
# Module-level helpers (testable as pure functions)
# ---------------------------------------------------------------------------


def _build_messages(
    title: str,
    prd_content: str,
    adr_content: str,
    existing_context: str | None,
) -> list[dict]:
    user_content = (
        f"Feature title: {title}\n\n"
        f"## PRD\n{prd_content}\n\n"
        f"## ADR\n{adr_content}"
    )
    if existing_context:
        user_content = (
            f"Relevant past implementations for reference:\n{existing_context}\n\n"
            f"---\n\n{user_content}"
        )
    return [{"role": "user", "content": user_content}]


def _parse_response(
    response: RouterResponse,
    feature_id: UUID,
    title: str,
    iteration: int,
) -> ImplementationResult:
    """
    Parse the LLM's JSON response into an ImplementationResult.
    Raises ValueError if the JSON is malformed or missing required fields.
    """
    raw = response.content.strip()

    # Strip any accidental code fences
    raw = re.sub(r"^```(?:json)?\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\n\nRaw:\n{raw[:500]}") from exc

    if "files" not in data or not isinstance(data["files"], list):
        raise ValueError("LLM response missing required 'files' list")

    files = [
        GeneratedFile(
            path=f["path"],
            description=f.get("description", ""),
            content=f["content"],
        )
        for f in data["files"]
        if "path" in f and "content" in f
    ]

    return ImplementationResult(
        feature_id=feature_id,
        title=title,
        summary=data.get("summary", ""),
        files=files,
        notes=data.get("notes", ""),
        estimated_test_coverage=data.get("estimated_test_coverage", ""),
        router_response=response,
        iteration=iteration,
    )


def _write_files(files: list[GeneratedFile]) -> None:
    """Write all generated files to disk, creating parent directories as needed."""
    for f in files:
        f.abs_path.parent.mkdir(parents=True, exist_ok=True)
        f.abs_path.write_text(f.content, encoding="utf-8")
        logger.debug("Wrote %s (%d bytes)", f.abs_path, len(f.content))
