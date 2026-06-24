"""
agents/implementation/agent.py — L3 Implementation Agent

Reads the ADR handoff from the Architecture Agent, retrieves related past
implementations via semantic search, generates code files via LLM, then:
  - Registers the implementation node in the traceability graph
  - Stores code embeddings for future semantic search
  - Hands off to the Testing Agent (L4)
  - Creates a human gate for high-complexity or breaking-change features
    (or auto-approves in Phase 2+)

Supports up to `max_regen_iterations` generation attempts (from workflow.yaml)
if an earlier attempt produces low-confidence output.

Run standalone:
    python -m agents.implementation.agent --feature-id <uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from uuid import UUID, uuid4

from agents.base import BaseAgent, AgentResult, AgentStatus
from agents.implementation.code_writer import CodeWriter, ImplementationResult
from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)


class ImplementationAgent(BaseAgent):
    """
    L3 — Implementation Agent

    Responsibilities:
      1. Receive ADR handoff from Architecture Agent
      2. Load PRD and ADR content from disk
      3. Semantic search for related past implementations
      4. Generate code files via LLM (implementation profile)
      5. Optionally regenerate if below confidence threshold
      6. Register implementation node in traceability graph
      7. Embed generated code for future semantic search
      8. Hand off to Testing Agent (L4)
      9. Create human gate (breaking changes, or always in Phase 1)
    """

    agent_name = "implementation"
    layer = 3

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
    ):
        super().__init__(store=store, router=router)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, feature_id: UUID, **kwargs) -> AgentResult:
        t0 = time.monotonic()
        stage_cfg = self._stage_config("implementation")
        max_regen = stage_cfg.get("max_regen_iterations", 3)

        # 1. Receive handoff from Architecture Agent
        handoff = await self.store.receive_handoff(
            feature_id=feature_id,
            agent="implementation",
        )
        if not handoff:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error="No handoff found from architecture agent — has architecture completed?",
            )

        adr_file = handoff.get("adr_file", "")
        title = handoff.get("title", "Untitled")
        is_breaking = handoff.get("is_breaking_change", False)
        complexity = handoff.get("complexity", "Unknown")
        redmine_id = handoff.get("redmine_id")

        # 2. Load ADR content (we need both ADR and PRD for context)
        adr_content = _read_file(adr_file)
        if not adr_content:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error=f"Could not read ADR file: {adr_file}",
            )

        # PRD is one directory up from ADR: docs/prds/{feature_id}.md
        prd_content = _read_prd_for_feature(feature_id)

        async with self.store.locked(f"feature:{feature_id}:implementation") as acquired:
            if not acquired:
                return AgentResult(
                    status=AgentStatus.SKIPPED,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    error="Implementation step already being processed by another run",
                )

            return await self._run_locked(
                feature_id=feature_id,
                title=title,
                prd_content=prd_content or "",
                adr_content=adr_content,
                is_breaking=is_breaking,
                complexity=complexity,
                redmine_id=redmine_id,
                max_regen=max_regen,
                t0=t0,
            )

    async def _run_locked(
        self,
        *,
        feature_id: UUID,
        title: str,
        prd_content: str,
        adr_content: str,
        is_breaking: bool,
        complexity: str,
        redmine_id: int | None,
        max_regen: int,
        t0: float,
    ) -> AgentResult:

        # 3. Semantic search for past implementations
        past_context = await self._fetch_past_context(title, adr_content)

        # 4. Log run start
        run_id = await self.begin(
            feature_id=feature_id,
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_summary=f"title={title!r}  breaking={is_breaking}  complexity={complexity}",
        )

        try:
            writer = CodeWriter(self._router)
            result: ImplementationResult | None = None

            # 5. Generate (with optional regen loop)
            for attempt in range(1, max_regen + 1):
                result = await writer.write(
                    feature_id=feature_id,
                    title=title,
                    prd_content=prd_content,
                    adr_content=adr_content,
                    existing_context=past_context,
                    iteration=attempt,
                )
                # In Phase 1 we always proceed; regen loop is wired for Phase 2+
                # when a code-review score is available before this point.
                break

            assert result is not None

            # 6. Advance feature status
            await self.store.advance_feature(feature_id, "implementation")

            # 7. Register implementation node in graph
            impl_id = uuid4()
            file_paths = [f.path for f in result.files]
            await self.store.graph.create_implementation(
                id=impl_id,
                feature_id=feature_id,
                status="draft",
            )

            # 8. Embed all generated files
            combined_code = "\n\n".join(
                f"# === {f.path} ===\n{f.content}" for f in result.files
            )
            await self.store.memorize(
                combined_code,
                feature_id=feature_id,
                artifact_type="implementation",
                artifact_id=str(impl_id),
            )

            # 9. Record decision
            await self.store.record_decision(
                run_id=run_id,
                feature_id=feature_id,
                agent=self.agent_name,
                decision_type="code_generated",
                summary=(
                    f"Generated {len(result.files)} file(s) for '{title}': "
                    + ", ".join(f.path for f in result.files)
                ),
                rationale=result.summary,
                outcome="pending_review",
            )

            # 10. Hand off to Testing Agent (L4)
            await self.store.handoff(
                feature_id=feature_id,
                from_agent="implementation",
                to_agent="testing",
                payload={
                    "impl_id": str(impl_id),
                    "title": title,
                    "file_paths": file_paths,
                    "is_breaking_change": is_breaking,
                    "complexity": complexity,
                    "summary": result.summary,
                    "redmine_id": redmine_id,
                },
            )

            duration_ms = int((time.monotonic() - t0) * 1000)
            cost = result.router_response.cost_usd

            # 11. Gate or succeed
            # Phase 1: always gate before testing begins (human reviews the code)
            stage_cfg = self._stage_config("implementation")
            always_gate = stage_cfg.get("always_gate", True)

            if always_gate or is_breaking or complexity == "High":
                trigger = (
                    "Phase 1: implementation always requires human review before testing"
                    if always_gate else
                    f"breaking={is_breaking} complexity={complexity}"
                )
                return await self.gate(
                    run_id=run_id,
                    feature_id=feature_id,
                    gate_type="implementation_review",
                    message=(
                        f"Review generated code for '{title}' ({len(result.files)} file(s)) "
                        f"before testing begins."
                    ),
                    trigger_reason=trigger,
                    payload={
                        "impl_id": str(impl_id),
                        "file_paths": file_paths,
                        "summary": result.summary,
                    },
                    cost_usd=cost,
                    duration_ms=duration_ms,
                )

            return await self.succeed(
                run_id=run_id,
                feature_id=feature_id,
                output={
                    "impl_id": str(impl_id),
                    "file_paths": file_paths,
                    "summary": result.summary,
                },
                cost_usd=cost,
                duration_ms=duration_ms,
                output_summary=f"Implementation auto-approved: {len(result.files)} file(s)",
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("ImplementationAgent failed for feature %s", feature_id)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_past_context(self, title: str, adr_content: str) -> str | None:
        """Semantic search for related past implementations."""
        try:
            query = f"{title}\n\n{adr_content[:500]}"
            hits = await self.store.remember(query, limit=3, artifact_type="implementation")
            if not hits:
                return None
            return "\n\n".join(
                f"[Past implementation excerpt, similarity={h['similarity']:.2f}]\n{h['content']}"
                for h in hits
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _read_file(path: str) -> str | None:
    """Read a file; return None on error."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read file %s: %s", path, exc)
        return None


def _read_prd_for_feature(feature_id: UUID) -> str | None:
    """Locate and read the PRD file for this feature."""
    from pathlib import Path as P
    prd_candidates = [
        P(__file__).parent.parent.parent / "docs" / "prds" / f"{feature_id}.md",
    ]
    for candidate in prd_candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def _detect_language(file_paths: list[str]) -> str:
    """Guess the primary language from file extensions."""
    ext_counts: dict[str, int] = {}
    for p in file_paths:
        ext = Path(p).suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    if not ext_counts:
        return "unknown"
    dominant = max(ext_counts, key=ext_counts.__getitem__)
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".sql": "sql",
    }.get(dominant, dominant.lstrip("."))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description="AI SDLC — Implementation Agent (L3)")
    parser.add_argument("--feature-id", "-f", required=True, type=UUID)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with ImplementationAgent() as agent:
        result = await agent.run(feature_id=args.feature_id)

    print(f"\n{'═' * 60}")
    print(f"  Status:     {result.status.value}")
    print(f"  Feature ID: {result.feature_id}")
    if result.gate_id:
        print(f"  Gate ID:    {result.gate_id}")
        print(f"  Next step:  Approve the code, then run the Testing Agent.")
    elif result.error:
        print(f"  Error:      {result.error}")
    print(f"  Cost:       ${result.cost_usd:.6f}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(_main())
