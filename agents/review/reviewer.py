"""
agents/review/reviewer.py — LLM code review + GitHub PR creation.

Produces a structured code review (score 0-100, narrative, issues list)
and creates a GitHub PR with the generated description.
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

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = """\
You are a principal engineer performing a thorough code review in an AI-powered SDLC pipeline.

Output a JSON object ONLY — no preamble, no explanation, no code fences.

Schema:
{
  "score": <integer 0-100>,
  "summary": "One-paragraph overall assessment.",
  "issues": [
    {
      "severity": "critical|high|medium|low|info",
      "file": "src/path/to/file.py",
      "line": <int or null>,
      "description": "What the issue is.",
      "suggestion": "How to fix it."
    }
  ],
  "strengths": ["List of things done well."],
  "must_fix": ["Issues that MUST be resolved before merge (critical/high only)."],
  "approved": <true|false>
}

Scoring guide:
  90-100: Excellent — approve immediately
  80-89:  Good — approve with minor notes
  70-79:  Acceptable — approve with fixes recommended
  < 70:   Needs work — do not approve, list must_fix items

Rules:
- Be specific: reference file names and line numbers where possible.
- "approved" must be true only if score >= 70 and must_fix is empty.
- Security issues are always critical or high severity.
- Consider: correctness, security, readability, test coverage, error handling.
"""

PR_SYSTEM_PROMPT = """\
You are a technical writer creating a GitHub Pull Request description.
Write a clear, professional PR description in Markdown.

Include:
## Summary
What this PR does and why.

## Changes
- List of key changes by file/module.

## Testing
How the changes were tested (unit tests, coverage, SAST results).

## Breaking Changes
YES/NO — describe any breaking changes.

## Checklist
- [ ] Tests pass
- [ ] Coverage meets threshold
- [ ] SAST scan clean
- [ ] ADR approved
- [ ] Documentation updated (if needed)

Be concise and factual. No marketing language.
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ReviewIssue:
    severity: str          # "critical" | "high" | "medium" | "low" | "info"
    file: str
    line: int | None
    description: str
    suggestion: str


@dataclass
class ReviewResult:
    score: float           # 0-100
    summary: str
    issues: list[ReviewIssue]
    strengths: list[str]
    must_fix: list[str]
    approved: bool
    router_response: RouterResponse


@dataclass
class PRResult:
    pr_number: int | None
    pr_url: str | None
    title: str
    body: str
    branch: str
    router_response: RouterResponse


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------


class CodeReviewer:
    """
    Runs LLM code review and creates the GitHub PR.

    Usage:
        reviewer = CodeReviewer(router)
        review = await reviewer.review(feature_id, title, impl_files, test_results)
        pr = await reviewer.create_pr(feature_id, title, branch, review, pr_meta)
    """

    def __init__(self, router: ModelRouter):
        self._router = router

    async def review(
        self,
        feature_id: UUID,
        title: str,
        impl_files: list[dict],          # [{"path": ..., "content": ...}]
        test_summary: dict,              # {passed, failed, coverage_pct, sast_findings}
        adr_summary: str | None = None,
        existing_context: str | None = None,
    ) -> ReviewResult:
        """Run LLM code review and return a structured result."""
        messages = _build_review_messages(
            title, impl_files, test_summary, adr_summary, existing_context
        )
        logger.info("Running code review for feature %s: %r", feature_id, title)

        response = await self._router.complete(
            profile="code_review",
            messages=messages,
            system=REVIEW_SYSTEM_PROMPT,
        )

        return _parse_review_response(response)

    async def generate_pr_description(
        self,
        feature_id: UUID,
        title: str,
        review: ReviewResult,
        test_summary: dict,
        is_breaking: bool,
        file_paths: list[str],
    ) -> str:
        """Generate a PR description body via LLM."""
        user_content = (
            f"Feature: {title}\n"
            f"Feature ID: {feature_id}\n\n"
            f"Review score: {review.score}/100\n"
            f"Review summary: {review.summary}\n\n"
            f"Changed files:\n" + "\n".join(f"  - {p}" for p in file_paths) + "\n\n"
            f"Test results: {test_summary.get('passed', 0)} passed, "
            f"{test_summary.get('failed', 0)} failed, "
            f"coverage={test_summary.get('coverage_pct', 0):.1f}%, "
            f"SAST findings={test_summary.get('sast_findings', 0)}\n\n"
            f"Breaking change: {'YES' if is_breaking else 'NO'}\n\n"
            f"Strengths:\n" + "\n".join(f"  - {s}" for s in review.strengths[:3])
        )

        response = await self._router.complete(
            profile="pr_description",
            messages=[{"role": "user", "content": user_content}],
            system=PR_SYSTEM_PROMPT,
        )
        return response.content.strip()

    async def create_github_pr(
        self,
        *,
        repo_name: str,
        title: str,
        body: str,
        branch: str,
        base_branch: str = "main",
        github_token: str | None = None,
    ) -> PRResult:
        """
        Create a GitHub PR using PyGithub.
        Returns PRResult with pr_number=None if GitHub is not configured.
        """
        # Lazy import — PyGithub is optional; skip if token not set
        if not github_token:
            logger.warning("GITHUB_TOKEN not set — skipping PR creation")
            return PRResult(
                pr_number=None,
                pr_url=None,
                title=title,
                body=body,
                branch=branch,
                router_response=_null_router_response(),
            )

        try:
            from github import Github, GithubException
            gh = Github(github_token)
            repo = gh.get_repo(repo_name)
            pr = repo.create_pull(
                title=title,
                body=body,
                head=branch,
                base=base_branch,
            )
            logger.info("Created GitHub PR #%d: %s", pr.number, pr.html_url)
            return PRResult(
                pr_number=pr.number,
                pr_url=pr.html_url,
                title=title,
                body=body,
                branch=branch,
                router_response=_null_router_response(),
            )
        except Exception as exc:
            logger.warning("GitHub PR creation failed: %s", exc)
            return PRResult(
                pr_number=None,
                pr_url=None,
                title=title,
                body=body,
                branch=branch,
                router_response=_null_router_response(),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_review_messages(
    title: str,
    impl_files: list[dict],
    test_summary: dict,
    adr_summary: str | None,
    existing_context: str | None,
) -> list[dict]:
    files_text = "\n\n".join(
        f"=== {f['path']} ===\n{f['content']}" for f in impl_files
    )
    user_content = (
        f"Feature: {title}\n\n"
        f"Test results: {test_summary.get('passed', 0)} passed, "
        f"{test_summary.get('failed', 0)} failed, "
        f"coverage={test_summary.get('coverage_pct', 0):.1f}%, "
        f"SAST findings={test_summary.get('sast_findings', 0)}\n\n"
    )
    if adr_summary:
        user_content += f"Architecture decisions:\n{adr_summary}\n\n"
    if existing_context:
        user_content += f"Past review patterns:\n{existing_context}\n\n"
    user_content += f"Implementation files:\n\n{files_text}"
    return [{"role": "user", "content": user_content}]


def _parse_review_response(response: RouterResponse) -> ReviewResult:
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for review: {exc}") from exc

    issues = [
        ReviewIssue(
            severity=i.get("severity", "info"),
            file=i.get("file", ""),
            line=i.get("line"),
            description=i.get("description", ""),
            suggestion=i.get("suggestion", ""),
        )
        for i in data.get("issues", [])
    ]

    return ReviewResult(
        score=float(data.get("score", 0)),
        summary=data.get("summary", ""),
        issues=issues,
        strengths=data.get("strengths", []),
        must_fix=data.get("must_fix", []),
        approved=bool(data.get("approved", False)),
        router_response=response,
    )


def _null_router_response() -> RouterResponse:
    return RouterResponse(
        content="",
        profile="pr_description",
        model_used="none",
        provider="none",
        was_fallback=False,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        duration_ms=0,
    )
