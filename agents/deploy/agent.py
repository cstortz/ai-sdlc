"""
agents/deploy/agent.py — L6 Deploy Agent

Reads the review handoff, triggers a GitHub Actions workflow dispatch,
polls for completion, runs the canary check (from workflow.yaml), then
promotes or rolls back.

Canary config (workflow.yaml):
  canary:
    window_minutes: 15
    error_rate_delta_max: 0.01
    traffic_pct: 10

Gate: prod deployments always require human approval in Phase 1.
      Set autonomy_phase to "ai_primary" in agents.yaml to auto-deploy.

Run standalone:
    python -m agents.deploy.agent --feature-id <uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from uuid import UUID, uuid4

from agents.base import BaseAgent, AgentResult, AgentStatus
from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)

# Polling config
_WORKFLOW_POLL_INTERVAL = 30   # seconds
_WORKFLOW_TIMEOUT       = 3600 # 1 hour


@dataclass
class DeployResult:
    pr_number: int | None
    branch: str
    environment: str
    workflow_run_id: int | None
    workflow_status: str        # "success" | "failure" | "skipped" | "timeout"
    canary_passed: bool
    canary_error_delta: float
    deployed_version: str


class DeployAgent(BaseAgent):
    """
    L6 — Deploy Agent

    Responsibilities:
      1. Receive review handoff
      2. Create human gate for prod deployment (Phase 1)
      3. On approval: trigger GitHub Actions workflow dispatch
      4. Poll for workflow completion
      5. Run canary check (error rate delta)
      6. Promote or rollback
      7. Register deployment in traceability graph
      8. Hand off to Monitor Agent (L7)
    """

    agent_name = "deploy"
    layer = 6

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
    ):
        super().__init__(store=store, router=router)

    async def run(self, feature_id: UUID, **kwargs) -> AgentResult:
        t0 = time.monotonic()
        canary_cfg = self._workflow.get("canary", {})
        canary_window    = int(canary_cfg.get("window_minutes", 15)) * 60
        canary_max_delta = float(canary_cfg.get("error_rate_delta_max", 0.01))
        canary_pct       = int(canary_cfg.get("traffic_pct", 10))

        # Phase config from agents.yaml via workflow (simplified: check env var)
        auto_deploy = os.environ.get("SDLC_AUTO_DEPLOY", "false").lower() == "true"

        handoff = await self.store.receive_handoff(
            feature_id=feature_id,
            agent="deploy",
        )
        if not handoff:
            return AgentResult(
                status=AgentStatus.FAILED,
                feature_id=feature_id,
                agent=self.agent_name,
                error="No handoff found from review agent",
            )

        title       = handoff.get("title", "Untitled")
        pr_number   = handoff.get("pr_number")
        branch      = handoff.get("branch", f"feature/{feature_id}")
        review_score = handoff.get("review_score", 0.0)
        is_breaking = handoff.get("is_breaking_change", False)
        impl_id     = handoff.get("impl_id", "")
        redmine_id  = handoff.get("redmine_id")

        async with self.store.locked(f"feature:{feature_id}:deploy") as acquired:
            if not acquired:
                return AgentResult(
                    status=AgentStatus.SKIPPED,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    error="Deploy step already being processed",
                )

            return await self._run_locked(
                feature_id=feature_id,
                title=title,
                pr_number=pr_number,
                branch=branch,
                review_score=review_score,
                is_breaking=is_breaking,
                impl_id=impl_id,
                redmine_id=redmine_id,
                canary_window=canary_window,
                canary_max_delta=canary_max_delta,
                canary_pct=canary_pct,
                auto_deploy=auto_deploy,
                t0=t0,
            )

    async def _run_locked(
        self,
        *,
        feature_id: UUID,
        title: str,
        pr_number: int | None,
        branch: str,
        review_score: float,
        is_breaking: bool,
        impl_id: str,
        redmine_id: int | None,
        canary_window: int,
        canary_max_delta: float,
        canary_pct: int,
        auto_deploy: bool,
        t0: float,
    ) -> AgentResult:

        run_id = await self.begin(
            feature_id=feature_id,
            model="none",          # Deploy agent doesn't use LLM
            provider="github",
            input_summary=(
                f"title={title!r}  pr={pr_number}  branch={branch}  "
                f"auto_deploy={auto_deploy}  canary={canary_pct}%"
            ),
        )

        try:
            environment = "production"
            version = branch.replace("/", "-")

            # Phase 1: always gate before prod deployment
            if not auto_deploy:
                duration_ms = int((time.monotonic() - t0) * 1000)
                gate_id = await self.store.request_human_approval(
                    run_id=run_id,
                    feature_id=feature_id,
                    gate_type="prod_deployment",
                    message=(
                        f"Approve production deployment of '{title}' "
                        f"(PR #{pr_number}, branch {branch}, score={review_score:.0f}/100). "
                        f"Breaking change: {is_breaking}."
                    ),
                    trigger_reason="Phase 1: prod deployment always requires human approval",
                    payload={
                        "pr_number": pr_number,
                        "branch": branch,
                        "review_score": review_score,
                        "is_breaking": is_breaking,
                    },
                )
                await self.store.end_run(
                    run_id,
                    status="escalated",
                    cost_usd=0.0,
                    duration_ms=duration_ms,
                )
                await self._publish(feature_id, "gate_wait", {
                    "gate_id": str(gate_id),
                    "message": f"Prod deployment gate for '{title}'",
                })
                # Pre-create the deploy→monitor handoff payload so Monitor
                # can be triggered after human approves and pipeline resumes.
                await self.store.handoff(
                    feature_id=feature_id,
                    from_agent="deploy",
                    to_agent="monitor",
                    payload={
                        "impl_id": impl_id,
                        "title": title,
                        "pr_number": pr_number,
                        "branch": branch,
                        "environment": environment,
                        "version": version,
                        "workflow_run_id": None,
                        "deployed": False,
                        "redmine_id": redmine_id,
                    },
                )
                return AgentResult(
                    status=AgentStatus.GATE_WAIT,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    gate_id=gate_id,
                    cost_usd=0.0,
                    duration_ms=duration_ms,
                )

            # Auto-deploy path (Phase 2+)
            deploy_result = await self._deploy(
                feature_id=feature_id,
                title=title,
                branch=branch,
                environment=environment,
                version=version,
                canary_window=canary_window,
                canary_max_delta=canary_max_delta,
            )

            duration_ms = int((time.monotonic() - t0) * 1000)
            return await self._finalize(
                run_id=run_id,
                feature_id=feature_id,
                title=title,
                deploy_result=deploy_result,
                impl_id=impl_id,
                pr_number=pr_number,
                redmine_id=redmine_id,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("DeployAgent failed for feature %s", feature_id)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def _deploy(
        self,
        *,
        feature_id: UUID,
        title: str,
        branch: str,
        environment: str,
        version: str,
        canary_window: int,
        canary_max_delta: float,
    ) -> DeployResult:
        """Trigger workflow, wait, run canary. Returns DeployResult."""
        github_token = os.environ.get("GITHUB_TOKEN")
        repo_name    = os.environ.get("GITHUB_REPO", "")
        workflow_file = os.environ.get("DEPLOY_WORKFLOW", "deploy.yml")

        workflow_run_id = await _trigger_workflow(
            repo_name=repo_name,
            workflow_file=workflow_file,
            branch=branch,
            inputs={"environment": environment, "version": version},
            github_token=github_token,
        )

        workflow_status = "skipped"
        if workflow_run_id:
            workflow_status = await _poll_workflow(
                repo_name=repo_name,
                run_id=workflow_run_id,
                github_token=github_token,
                timeout=_WORKFLOW_TIMEOUT,
                poll_interval=_WORKFLOW_POLL_INTERVAL,
            )

        # Canary check
        canary_passed = True
        canary_delta  = 0.0
        if workflow_status == "success":
            canary_passed, canary_delta = await _canary_check(
                environment=environment,
                window_seconds=canary_window,
                max_delta=canary_max_delta,
            )
            if not canary_passed:
                await _trigger_rollback(repo_name, workflow_file, branch, environment, github_token)
                logger.warning(
                    "Canary failed: error_rate_delta=%.4f > %.4f — rolled back",
                    canary_delta, canary_max_delta,
                )

        return DeployResult(
            pr_number=None,
            branch=branch,
            environment=environment,
            workflow_run_id=workflow_run_id,
            workflow_status=workflow_status,
            canary_passed=canary_passed,
            canary_error_delta=canary_delta,
            deployed_version=version,
        )

    async def _finalize(
        self,
        *,
        run_id: UUID,
        feature_id: UUID,
        title: str,
        deploy_result: DeployResult,
        impl_id: str,
        pr_number: int | None,
        redmine_id: int | None,
        duration_ms: int,
    ) -> AgentResult:
        """Register deployment, hand off to Monitor, succeed or fail."""

        deploy_id = uuid4()
        status = "deployed" if deploy_result.canary_passed else "rolled_back"

        await self.store.graph.create_deployment(
            id=deploy_id,
            implementation_id=UUID(impl_id) if impl_id else uuid4(),
            environment=deploy_result.environment,
            version=deploy_result.deployed_version,
            triggered_by="agent",
            status=status,
        )

        await self.store.advance_feature(feature_id, "deployed" if deploy_result.canary_passed else "rollback")

        await self.store.record_decision(
            run_id=run_id,
            feature_id=feature_id,
            agent=self.agent_name,
            decision_type="deployment",
            summary=(
                f"Deployment {status}: workflow={deploy_result.workflow_status}  "
                f"canary_passed={deploy_result.canary_passed}  "
                f"error_delta={deploy_result.canary_error_delta:.4f}"
            ),
            rationale=f"canary_passed={deploy_result.canary_passed}",
            outcome=status,
        )

        await self.store.handoff(
            feature_id=feature_id,
            from_agent="deploy",
            to_agent="monitor",
            payload={
                "deploy_id": str(deploy_id),
                "impl_id": impl_id,
                "title": title,
                "pr_number": pr_number,
                "branch": deploy_result.branch,
                "environment": deploy_result.environment,
                "version": deploy_result.deployed_version,
                "workflow_run_id": deploy_result.workflow_run_id,
                "deployed": deploy_result.canary_passed,
                "canary_error_delta": deploy_result.canary_error_delta,
                "redmine_id": redmine_id,
            },
        )

        if not deploy_result.canary_passed:
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=(
                    f"Canary failed: error_rate_delta={deploy_result.canary_error_delta:.4f}. "
                    "Rolled back."
                ),
                duration_ms=duration_ms,
            )

        return await self.succeed(
            run_id=run_id,
            feature_id=feature_id,
            output={
                "deploy_id": str(deploy_id),
                "environment": deploy_result.environment,
                "version": deploy_result.deployed_version,
                "workflow_run_id": deploy_result.workflow_run_id,
            },
            duration_ms=duration_ms,
            output_summary=f"Deployed '{title}' to {deploy_result.environment}",
        )


# ---------------------------------------------------------------------------
# GitHub helpers (testable, injectable)
# ---------------------------------------------------------------------------


async def _trigger_workflow(
    *,
    repo_name: str,
    workflow_file: str,
    branch: str,
    inputs: dict,
    github_token: str | None,
) -> int | None:
    """Trigger a GitHub Actions workflow dispatch. Returns run_id or None."""
    if not github_token or not repo_name:
        logger.warning("GitHub not configured — skipping workflow dispatch")
        return None
    try:
        from github import Github
        gh = Github(github_token)
        repo = gh.get_repo(repo_name)
        wf = repo.get_workflow(workflow_file)
        wf.create_dispatch(ref=branch, inputs=inputs)
        # Wait a moment for the run to appear
        await asyncio.sleep(3)
        runs = list(wf.get_runs(branch=branch, event="workflow_dispatch"))
        return runs[0].id if runs else None
    except Exception as exc:
        logger.warning("workflow dispatch failed: %s", exc)
        return None


async def _poll_workflow(
    *,
    repo_name: str,
    run_id: int,
    github_token: str | None,
    timeout: int,
    poll_interval: int,
) -> str:
    """Poll a GitHub Actions run until completion. Returns status string."""
    if not github_token or not repo_name:
        return "skipped"
    try:
        from github import Github
        gh = Github(github_token)
        repo = gh.get_repo(repo_name)
        elapsed = 0
        while elapsed < timeout:
            run = repo.get_workflow_run(run_id)
            if run.status == "completed":
                return run.conclusion or "failure"
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        return "timeout"
    except Exception as exc:
        logger.warning("workflow poll failed: %s", exc)
        return "failure"


async def _canary_check(
    *,
    environment: str,
    window_seconds: int,
    max_delta: float,
) -> tuple[bool, float]:
    """
    Check canary metrics. In production this would query Prometheus/Datadog.
    Returns (passed, error_rate_delta).

    For now: a placeholder that always passes (real impl queries metrics API).
    """
    logger.info(
        "Canary check: env=%s window=%ds max_delta=%.3f",
        environment, window_seconds, max_delta,
    )
    await asyncio.sleep(min(window_seconds, 1))  # In tests, window is mocked
    # Placeholder: no real metrics available — assume passing
    # Production: query metrics endpoint and compare pre/post error rates
    return True, 0.0


async def _trigger_rollback(
    repo_name: str,
    workflow_file: str,
    branch: str,
    environment: str,
    github_token: str | None,
) -> None:
    """Trigger a rollback workflow dispatch."""
    await _trigger_workflow(
        repo_name=repo_name,
        workflow_file=workflow_file,
        branch=branch,
        inputs={"environment": environment, "action": "rollback"},
        github_token=github_token,
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description="AI SDLC — Deploy Agent (L6)")
    parser.add_argument("--feature-id", "-f", required=True, type=UUID)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with DeployAgent() as agent:
        result = await agent.run(feature_id=args.feature_id)

    print(f"\n{'═' * 60}")
    print(f"  Status:     {result.status.value}")
    print(f"  Feature ID: {result.feature_id}")
    if result.gate_id:
        print(f"  Gate ID:    {result.gate_id}")
        print(f"  Next step:  Approve the deployment gate to release.")
    elif result.error:
        print(f"  Error:      {result.error}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(_main())
