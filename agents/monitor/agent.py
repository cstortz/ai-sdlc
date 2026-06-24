"""
agents/monitor/agent.py — L7 Monitor Agent

Closes the pipeline loop. Triggered after deployment, then continues as a
periodic triage agent for the lifetime of the feature.

Responsibilities:
  1. Receive deploy handoff (feature context)
  2. Check for open incidents linked to this deployment
  3. Classify severity via LLM (monitor_triage profile)
  4. Auto-resolve LOW incidents; escalate MEDIUM+ to human gate
  5. Detect recurring incident patterns via graph
  6. Register incidents in traceability graph
  7. Publish alerts to Redis incident channel

Scheduled mode:
  Run with --scan-all to triage all open incidents across all features.
  Designed for use with the Cowork schedule skill (cron every 15min).

Run standalone:
    python -m agents.monitor.agent --feature-id <uuid>
    python -m agents.monitor.agent --scan-all
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from uuid import UUID, uuid4

from agents.base import BaseAgent, AgentResult, AgentStatus
from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)

_AUTO_RESOLVE_SEVERITY = {"low"}   # Severities the agent resolves without human


@dataclass
class IncidentClassification:
    severity: str          # "low" | "medium" | "high" | "critical"
    summary: str
    recommended_action: str
    is_recurring: bool
    recurrence_count: int


class MonitorAgent(BaseAgent):
    """
    L7 — Monitor Agent

    Completes the pipeline loop and provides ongoing triage.
    """

    agent_name = "monitor"
    layer = 7

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
    ):
        super().__init__(store=store, router=router)

    async def run(self, feature_id: UUID, **kwargs) -> AgentResult:
        """
        Primary run: process the deploy handoff and check for incidents.
        """
        t0 = time.monotonic()

        handoff = await self.store.receive_handoff(
            feature_id=feature_id,
            agent="monitor",
        )
        if not handoff:
            # No handoff = called as a scheduled scan for this feature
            return await self._scan_feature(feature_id=feature_id, t0=t0)

        title       = handoff.get("title", "Untitled")
        deploy_id   = handoff.get("deploy_id")
        environment = handoff.get("environment", "production")
        version     = handoff.get("version", "unknown")
        deployed    = handoff.get("deployed", False)
        redmine_id  = handoff.get("redmine_id")

        async with self.store.locked(f"feature:{feature_id}:monitor") as acquired:
            if not acquired:
                return AgentResult(
                    status=AgentStatus.SKIPPED,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    error="Monitor step already running",
                )

            return await self._run_post_deploy(
                feature_id=feature_id,
                title=title,
                deploy_id=deploy_id,
                environment=environment,
                version=version,
                deployed=deployed,
                redmine_id=redmine_id,
                t0=t0,
            )

    async def _run_post_deploy(
        self,
        *,
        feature_id: UUID,
        title: str,
        deploy_id: str | None,
        environment: str,
        version: str,
        deployed: bool,
        redmine_id: int | None,
        t0: float,
    ) -> AgentResult:
        """Post-deployment monitoring: check for immediate incidents."""

        run_id = await self.begin(
            feature_id=feature_id,
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_summary=(
                f"title={title!r}  environment={environment}  "
                f"version={version}  deployed={deployed}"
            ),
        )

        try:
            await self.store.advance_feature(feature_id, "monitoring")

            # Check for open incidents linked to this deployment
            open_incidents = await self.store.open_incidents()
            feature_incidents = [
                i for i in open_incidents
                if str(i.get("feature_id", "")) == str(feature_id)
            ]

            # Classify and triage any found incidents
            escalated_count = 0
            resolved_count  = 0

            for incident in feature_incidents:
                classification = await self._classify_incident(incident, feature_id)
                action = await self._triage_incident(
                    incident=incident,
                    classification=classification,
                    feature_id=feature_id,
                    deploy_id=deploy_id,
                    run_id=run_id,
                )
                if action == "escalated":
                    escalated_count += 1
                elif action == "resolved":
                    resolved_count += 1

            # Check recurring patterns
            recurring = await self.store.recurring_incidents()
            if recurring:
                await self._publish_alert(
                    feature_id=feature_id,
                    message=(
                        f"Recurring incident pattern detected: "
                        f"{len(recurring)} pattern(s) in {environment}"
                    ),
                    severity="medium",
                )

            # Record completion
            await self.store.record_decision(
                run_id=run_id,
                feature_id=feature_id,
                agent=self.agent_name,
                decision_type="monitoring_complete",
                summary=(
                    f"Post-deploy monitoring for '{title}' in {environment}: "
                    f"{len(feature_incidents)} incident(s), "
                    f"{resolved_count} resolved, {escalated_count} escalated"
                ),
                rationale=f"deployed={deployed}  version={version}",
                outcome="monitoring_active",
            )

            duration_ms = int((time.monotonic() - t0) * 1000)

            if escalated_count > 0:
                # Gate for human to review escalated incidents
                gate_id = await self.store.request_human_approval(
                    run_id=run_id,
                    feature_id=feature_id,
                    gate_type="incident_review",
                    message=(
                        f"{escalated_count} incident(s) require human review "
                        f"in {environment} for '{title}'."
                    ),
                    trigger_reason=f"{escalated_count} MEDIUM+ severity incidents",
                    payload={
                        "deploy_id": deploy_id,
                        "environment": environment,
                        "escalated_count": escalated_count,
                        "resolved_count": resolved_count,
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
                    "message": f"Incident review gate for '{title}'",
                })
                return AgentResult(
                    status=AgentStatus.GATE_WAIT,
                    feature_id=feature_id,
                    agent=self.agent_name,
                    gate_id=gate_id,
                    duration_ms=duration_ms,
                )

            return await self.succeed(
                run_id=run_id,
                feature_id=feature_id,
                output={
                    "environment": environment,
                    "version": version,
                    "incidents_found": len(feature_incidents),
                    "resolved": resolved_count,
                    "escalated": escalated_count,
                },
                duration_ms=duration_ms,
                output_summary=(
                    f"Pipeline complete: '{title}' deployed to {environment}. "
                    f"{resolved_count} incident(s) auto-resolved."
                ),
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("MonitorAgent failed for feature %s", feature_id)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def _scan_feature(self, *, feature_id: UUID, t0: float) -> AgentResult:
        """Scheduled scan: triage open incidents for a specific feature."""
        run_id = await self.begin(
            feature_id=feature_id,
            model="claude-sonnet-4-6",
            provider="anthropic",
            input_summary=f"scheduled_scan  feature={feature_id}",
        )
        try:
            open_incidents = await self.store.open_incidents()
            feature_incidents = [
                i for i in open_incidents
                if str(i.get("feature_id", "")) == str(feature_id)
            ]

            resolved = 0
            for incident in feature_incidents:
                classification = await self._classify_incident(incident, feature_id)
                action = await self._triage_incident(
                    incident=incident,
                    classification=classification,
                    feature_id=feature_id,
                    deploy_id=None,
                    run_id=run_id,
                )
                if action == "resolved":
                    resolved += 1

            duration_ms = int((time.monotonic() - t0) * 1000)
            return await self.succeed(
                run_id=run_id,
                feature_id=feature_id,
                output={
                    "scanned": len(feature_incidents),
                    "resolved": resolved,
                },
                duration_ms=duration_ms,
                output_summary=f"Scan: {resolved}/{len(feature_incidents)} incidents resolved",
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return await self.fail(
                run_id=run_id,
                feature_id=feature_id,
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def scan_all(self) -> dict[str, int]:
        """
        Triage ALL open incidents across all features.
        Designed for the scheduled cron job (every 15min).
        Returns {resolved: int, escalated: int, total: int}.
        """
        open_incidents = await self.store.open_incidents()
        recurring      = await self.store.recurring_incidents()

        resolved_total  = 0
        escalated_total = 0

        for incident in open_incidents:
            feature_id_str = incident.get("feature_id")
            try:
                fid = UUID(feature_id_str) if feature_id_str else uuid4()
            except ValueError:
                fid = uuid4()

            classification = await self._classify_incident(incident, fid)
            severity = classification.severity.lower()

            if severity in _AUTO_RESOLVE_SEVERITY:
                await self.store.graph.update_node_status(
                    "Incident",
                    UUID(str(incident.get("id", uuid4()))),
                    "resolved",
                )
                resolved_total += 1
            else:
                escalated_total += 1
                await self.store.cache.publish_incident(
                    incident_id=uuid4(),
                    severity=severity,
                    title=classification.summary,
                )

        if recurring:
            logger.warning(
                "Recurring incident patterns detected: %d pattern(s)", len(recurring)
            )

        return {
            "total": len(open_incidents),
            "resolved": resolved_total,
            "escalated": escalated_total,
            "recurring_patterns": len(recurring),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _classify_incident(
        self,
        incident: dict,
        feature_id: UUID,
    ) -> IncidentClassification:
        """Use LLM to classify incident severity and recommend action."""
        title    = incident.get("title", "Unknown incident")
        desc     = incident.get("description", "")
        severity = incident.get("severity", "medium")

        # Check for recurring pattern
        try:
            related = await self.store.graph.get_related_decisions(feature_id)
            recurring_count = sum(
                1 for d in related
                if title.lower() in str(d.get("summary", "")).lower()
            )
        except Exception:
            recurring_count = 0

        # Use LLM for analysis on MEDIUM+ incidents
        if severity.lower() in ("medium", "high", "critical"):
            try:
                response = await self._router.complete(
                    profile="incident_analysis",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Incident: {title}\n"
                            f"Description: {desc}\n"
                            f"Severity: {severity}\n"
                            f"Recurrence: {recurring_count} similar past incidents\n\n"
                            "Classify this incident and recommend action in one sentence each."
                        ),
                    }],
                    system=(
                        "You are a site reliability engineer. "
                        "Respond with: SEVERITY: <low|medium|high|critical>\n"
                        "ACTION: <recommended action>"
                    ),
                )
                content = response.content
                sev_match = __import__("re").search(
                    r"SEVERITY:\s*(\w+)", content, __import__("re").IGNORECASE
                )
                act_match = __import__("re").search(
                    r"ACTION:\s*(.+)", content, __import__("re").IGNORECASE
                )
                severity = sev_match.group(1).lower() if sev_match else severity
                action   = act_match.group(1).strip() if act_match else "Investigate"
            except Exception:
                action = "Investigate"
        else:
            action = "Auto-resolve (low severity)"

        return IncidentClassification(
            severity=severity,
            summary=title,
            recommended_action=action,
            is_recurring=recurring_count > 0,
            recurrence_count=recurring_count,
        )

    async def _triage_incident(
        self,
        *,
        incident: dict,
        classification: IncidentClassification,
        feature_id: UUID,
        deploy_id: str | None,
        run_id: UUID,
    ) -> str:
        """
        Act on incident classification.
        Returns "resolved", "escalated", or "skipped".
        """
        severity = classification.severity.lower()
        incident_id_str = incident.get("id", str(uuid4()))

        try:
            incident_uuid = UUID(incident_id_str)
        except ValueError:
            incident_uuid = uuid4()

        if severity in _AUTO_RESOLVE_SEVERITY:
            await self.store.graph.update_node_status("Incident", incident_uuid, "resolved")
            logger.info(
                "Auto-resolved LOW incident %s: %s",
                incident_id_str, classification.summary,
            )
            return "resolved"

        # Escalate MEDIUM+
        await self._publish_alert(
            feature_id=feature_id,
            message=(
                f"[{severity.upper()}] {classification.summary}: "
                f"{classification.recommended_action}"
            ),
            severity=severity,
        )
        logger.warning(
            "Escalated %s incident %s: %s",
            severity, incident_id_str, classification.summary,
        )
        return "escalated"

    async def _publish_alert(
        self,
        feature_id: UUID,
        message: str,
        severity: str,
    ) -> None:
        """Publish to Redis incidents channel."""
        try:
            await self.store.cache.publish_incident(
                incident_id=uuid4(),
                severity=severity,
                title=message,
            )
        except Exception as exc:
            logger.warning("Failed to publish incident alert: %s", exc)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description="AI SDLC — Monitor Agent (L7)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--feature-id", "-f", type=UUID, help="Feature UUID (post-deploy)")
    group.add_argument("--scan-all", action="store_true", help="Triage all open incidents")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    async with MonitorAgent() as agent:
        if args.scan_all:
            summary = await agent.scan_all()
            print(f"\n  Scan complete: {summary}")
        else:
            result = await agent.run(feature_id=args.feature_id)
            print(f"\n  Status: {result.status.value}  Feature: {result.feature_id}")
            if result.error:
                print(f"  Error:  {result.error}")


if __name__ == "__main__":
    asyncio.run(_main())
