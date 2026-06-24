"""
agents/base.py — BaseAgent

All pipeline agents subclass this. It handles:
  - ContextStore lifecycle (connect/close via context manager)
  - ModelRouter wiring
  - Workflow config loading (config/workflow.yaml)
  - Run audit logging (begin_run / end_run)
  - Human gate creation and gate-wait loop
  - Redis event publishing (completion + error events)

Subclasses implement:
  async def run(self, feature_id: UUID, **kwargs) -> AgentResult
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from context_store import ContextStore
from router import ModelRouter

logger = logging.getLogger(__name__)

_WORKFLOW_CONFIG_PATH = Path(__file__).parent.parent / "config" / "workflow.yaml"
_MODELS_CONFIG_PATH   = Path(__file__).parent.parent / "config" / "models.yaml"

# Redis event channel — all agents publish here on completion
CHANNEL_PIPELINE = "sdlc:pipeline"


class AgentStatus(str, Enum):
    SUCCESS    = "success"
    GATE_WAIT  = "gate_wait"    # paused at a human gate
    FAILED     = "failed"
    SKIPPED    = "skipped"


@dataclass
class AgentResult:
    """Returned by every agent's run() method."""
    status: AgentStatus
    feature_id: UUID
    agent: str
    output: dict = field(default_factory=dict)       # agent-specific output payload
    gate_id: UUID | None = None                       # set when status == GATE_WAIT
    error: str | None = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    confidence: float | None = None


class BaseAgent(ABC):
    """
    Abstract base for all SDLC pipeline agents.

    Usage (from pipeline.py or standalone):
        agent = IntakeAgent()
        async with agent:
            result = await agent.run(feature_id=fid, prompt="Build a login page")
    """

    # Subclasses set these
    agent_name: str = "base"
    layer: int = 0

    def __init__(
        self,
        store: ContextStore | None = None,
        router: ModelRouter | None = None,
    ):
        # Dependency injection — pipeline passes shared instances; tests pass mocks
        self._store  = store
        self._router = router or ModelRouter(config_path=_MODELS_CONFIG_PATH)
        self._workflow = self._load_workflow_config()
        self._owns_store = store is None  # True = we created it, we close it

    async def __aenter__(self) -> "BaseAgent":
        if self._store is None:
            self._store = ContextStore()
            await self._store.connect()
        return self

    async def __aexit__(self, *_) -> None:
        if self._owns_store and self._store:
            await self._store.close()

    @property
    def store(self) -> ContextStore:
        if not self._store:
            raise RuntimeError(f"{self.agent_name}: not connected. Use 'async with agent:'")
        return self._store

    @property
    def router(self) -> ModelRouter:
        return self._router

    @property
    def workflow(self) -> dict:
        return self._workflow

    # ------------------------------------------------------------------
    # Core interface — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, feature_id: UUID, **kwargs) -> AgentResult:
        """Execute the agent's work. Must be called inside 'async with agent:'."""

    # ------------------------------------------------------------------
    # Audit helpers — called by subclasses
    # ------------------------------------------------------------------

    async def begin(
        self,
        feature_id: UUID,
        model: str,
        provider: str,
        was_fallback: bool = False,
        input_summary: str | None = None,
    ) -> UUID:
        """Log run start and return run_id."""
        return await self.store.begin_run(
            feature_id=feature_id,
            agent=self.agent_name,
            layer=self.layer,
            model=model,
            provider=provider,
            was_fallback=was_fallback,
            input_summary=input_summary,
        )

    async def succeed(
        self,
        run_id: UUID,
        feature_id: UUID,
        output: dict,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        confidence: float | None = None,
        output_summary: str | None = None,
    ) -> AgentResult:
        """Mark run complete and publish pipeline event."""
        await self.store.end_run(
            run_id,
            status="completed",
            output_summary=output_summary or str(output)[:200],
            confidence=confidence,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )
        await self._publish(feature_id, "completed", output)
        return AgentResult(
            status=AgentStatus.SUCCESS,
            feature_id=feature_id,
            agent=self.agent_name,
            output=output,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            confidence=confidence,
        )

    async def gate(
        self,
        run_id: UUID,
        feature_id: UUID,
        *,
        gate_type: str,
        message: str,
        trigger_reason: str = "",
        payload: dict | None = None,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ) -> AgentResult:
        """Create a human gate, mark run as escalated, publish gate event."""
        gate_id = await self.store.request_human_approval(
            run_id=run_id,
            feature_id=feature_id,
            gate_type=gate_type,
            message=message,
            trigger_reason=trigger_reason or gate_type,
            payload=payload,
        )
        await self.store.end_run(
            run_id,
            status="escalated",
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )
        await self._publish(feature_id, "gate_wait", {"gate_id": str(gate_id), "message": message})
        logger.info("%s: human gate created — %s", self.agent_name, message)
        return AgentResult(
            status=AgentStatus.GATE_WAIT,
            feature_id=feature_id,
            agent=self.agent_name,
            gate_id=gate_id,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

    async def fail(
        self,
        run_id: UUID,
        feature_id: UUID,
        error: str,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ) -> AgentResult:
        """Mark run failed and publish error event."""
        await self.store.end_run(
            run_id,
            status="failed",
            error_message=error,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )
        await self._publish(feature_id, "failed", {"error": error})
        logger.error("%s: failed — %s", self.agent_name, error)
        return AgentResult(
            status=AgentStatus.FAILED,
            feature_id=feature_id,
            agent=self.agent_name,
            error=error,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Gate-wait polling (used by pipeline.py)
    # ------------------------------------------------------------------

    async def wait_for_gate(
        self,
        gate_id: UUID,
        feature_id: UUID,
        *,
        poll_interval_seconds: int = 30,
        timeout_hours: float | None = None,
    ) -> bool:
        """
        Poll until the human gate is resolved.
        Returns True if approved, False if rejected or timed out.
        """
        import json
        timeout_s = (timeout_hours * 3600) if timeout_hours else None
        elapsed = 0.0
        logger.info(
            "%s: waiting for gate %s (poll every %ds)",
            self.agent_name, gate_id, poll_interval_seconds,
        )
        while True:
            gates = await self.store.pending_gates(feature_id)
            this_gate = next((g for g in gates if str(g["id"]) == str(gate_id)), None)
            if this_gate is None:
                # Gate resolved — check status from DB directly
                break
            await asyncio.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds
            if timeout_s and elapsed >= timeout_s:
                logger.warning("%s: gate %s timed out", self.agent_name, gate_id)
                return False
        return True  # resolved; caller checks approved/rejected status from DB

    # ------------------------------------------------------------------
    # Workflow config helpers
    # ------------------------------------------------------------------

    def _stage_config(self, stage: str) -> dict:
        return self._workflow.get("stages", {}).get(stage, {})

    def _threshold(self, key: str, default: Any = None) -> Any:
        return self._workflow.get("thresholds", {}).get(key, default)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _publish(self, feature_id: UUID, event: str, payload: dict) -> None:
        """Publish a pipeline event to Redis (fire-and-forget; swallow errors)."""
        try:
            import json
            envelope = {
                "agent": self.agent_name,
                "layer": self.layer,
                "feature_id": str(feature_id),
                "event": event,
                **payload,
            }
            await self.store.cache.client.publish(
                CHANNEL_PIPELINE, json.dumps(envelope)
            )
        except Exception as exc:
            # Don't let a Redis failure break the pipeline
            logger.warning("%s: failed to publish pipeline event: %s", self.agent_name, exc)

    @staticmethod
    def _load_workflow_config() -> dict:
        try:
            with open(_WORKFLOW_CONFIG_PATH) as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("workflow.yaml not found — using empty config")
            return {}
