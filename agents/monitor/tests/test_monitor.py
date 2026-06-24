"""
agents/monitor/tests/test_monitor.py

Unit tests for the Monitor Agent.

Run: pytest agents/monitor/tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agents.base import AgentStatus
from agents.monitor.agent import MonitorAgent, IncidentClassification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _incident(severity: str = "low", title: str = "500 errors") -> dict:
    return {
        "id": str(uuid4()),
        "feature_id": str(uuid4()),
        "severity": severity,
        "title": title,
        "description": f"Multiple {severity} errors in auth service.",
    }


def _mock_store(fid: UUID, handoff: dict | None = None) -> MagicMock:
    store = MagicMock()
    store.receive_handoff = AsyncMock(return_value=handoff or {
        "deploy_id": str(uuid4()),
        "impl_id": str(uuid4()),
        "title": "User Auth",
        "pr_number": 42,
        "branch": "feature/abc",
        "environment": "production",
        "version": "feature-abc",
        "deployed": True,
        "canary_error_delta": 0.001,
        "redmine_id": None,
    })
    store.advance_feature = AsyncMock()
    store.graph = MagicMock()
    store.graph.update_node_status = AsyncMock()
    store.graph.get_related_decisions = AsyncMock(return_value=[])
    store.open_incidents = AsyncMock(return_value=[])
    store.recurring_incidents = AsyncMock(return_value=[])
    store.memorize = AsyncMock(return_value=1)
    store.begin_run = AsyncMock(return_value=uuid4())
    store.end_run = AsyncMock()
    store.request_human_approval = AsyncMock(return_value=uuid4())
    store.record_decision = AsyncMock(return_value=uuid4())
    store.handoff = AsyncMock(return_value=uuid4())
    store.remember = AsyncMock(return_value=[])
    store.cache = MagicMock()
    store.cache.client = MagicMock()
    store.cache.client.publish = AsyncMock(return_value=1)
    store.cache.publish_incident = AsyncMock()

    @asynccontextmanager
    async def _locked(*a, **kw):
        yield True

    store.locked = _locked
    return store


def _mock_router(classification_text: str = "SEVERITY: low\nACTION: Monitor") -> MagicMock:
    from router import RouterResponse
    router = MagicMock()
    router.complete = AsyncMock(return_value=RouterResponse(
        content=classification_text,
        profile="incident_analysis",
        model_used="claude-sonnet-4-6",
        provider="anthropic",
        was_fallback=False,
        input_tokens=50,
        output_tokens=20,
        cost_usd=0.0001,
        duration_ms=200,
    ))
    return router


def _make_agent(fid: UUID | None = None, handoff: dict | None = None):
    fid = fid or uuid4()
    store = _mock_store(fid, handoff)
    router = _mock_router()
    agent = MonitorAgent(store=store, router=router)
    return agent, store, fid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMonitorAgent:

    @pytest.mark.asyncio
    async def test_run_succeeds_with_no_incidents(self):
        agent, store, fid = _make_agent()
        result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_run_auto_resolves_low_incidents(self):
        agent, store, fid = _make_agent()
        incident = _incident("low")
        incident["feature_id"] = str(fid)
        store.open_incidents = AsyncMock(return_value=[incident])

        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.SUCCESS
        # Low severity → auto-resolved, no gate
        store.request_human_approval.assert_not_called()
        store.graph.update_node_status.assert_called()

    @pytest.mark.asyncio
    async def test_run_escalates_medium_incidents(self):
        agent, store, fid = _make_agent()
        incident = _incident("medium", "DB connection errors")
        incident["feature_id"] = str(fid)
        store.open_incidents = AsyncMock(return_value=[incident])
        # LLM returns medium classification
        agent._router = _mock_router("SEVERITY: medium\nACTION: Investigate DB pool")

        result = await agent.run(feature_id=fid)

        assert result.status == AgentStatus.GATE_WAIT
        store.request_human_approval.assert_called_once()
        store.cache.publish_incident.assert_called()

    @pytest.mark.asyncio
    async def test_run_fails_without_handoff_falls_back_to_scan(self):
        """No handoff → treated as a scan run → succeeds."""
        agent, store, fid = _make_agent()
        store.receive_handoff = AsyncMock(return_value=None)
        store.open_incidents = AsyncMock(return_value=[])

        result = await agent.run(feature_id=fid)

        # _scan_feature is called → succeeds (no incidents)
        assert result.status == AgentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_run_skipped_when_lock_held(self):
        agent, store, fid = _make_agent()

        @asynccontextmanager
        async def _no(*a, **kw):
            yield False

        store.locked = _no
        result = await agent.run(feature_id=fid)
        assert result.status == AgentStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_run_records_decision(self):
        agent, store, fid = _make_agent()
        await agent.run(feature_id=fid)
        store.record_decision.assert_called_once()
        kwargs = store.record_decision.call_args[1]
        assert kwargs["agent"] == "monitor"
        assert kwargs["decision_type"] == "monitoring_complete"

    @pytest.mark.asyncio
    async def test_scan_all_resolves_low_auto(self):
        fid = uuid4()
        agent, store, _ = _make_agent(fid)
        incident = _incident("low")
        store.open_incidents = AsyncMock(return_value=[incident])
        store.recurring_incidents = AsyncMock(return_value=[])

        summary = await agent.scan_all()

        assert summary["total"] == 1
        assert summary["resolved"] == 1
        assert summary["escalated"] == 0

    @pytest.mark.asyncio
    async def test_scan_all_escalates_high(self):
        fid = uuid4()
        agent, store, _ = _make_agent(fid)
        # Router must return HIGH so classification isn't overridden to LOW
        agent._router = _mock_router("SEVERITY: high\nACTION: Investigate immediately")
        incident = _incident("high", "Memory leak")
        store.open_incidents = AsyncMock(return_value=[incident])
        store.recurring_incidents = AsyncMock(return_value=[])

        summary = await agent.scan_all()

        assert summary["total"] == 1
        assert summary["escalated"] == 1
        assert summary["resolved"] == 0
        store.cache.publish_incident.assert_called()

    @pytest.mark.asyncio
    async def test_scan_all_reports_recurring_patterns(self):
        fid = uuid4()
        agent, store, _ = _make_agent(fid)
        store.open_incidents = AsyncMock(return_value=[])
        store.recurring_incidents = AsyncMock(return_value=[{"pattern": "DB timeout"}])

        summary = await agent.scan_all()

        assert summary["recurring_patterns"] == 1

    @pytest.mark.asyncio
    async def test_classify_incident_low_no_llm_call(self):
        """Low severity incidents skip LLM classification."""
        agent, store, fid = _make_agent()
        store.graph.get_related_decisions = AsyncMock(return_value=[])
        incident = _incident("low")

        classification = await agent._classify_incident(incident, fid)

        assert classification.severity == "low"
        # LLM should NOT be called for low severity
        agent._router.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_incident_high_calls_llm(self):
        """High severity incidents use LLM for analysis."""
        agent, store, fid = _make_agent()
        agent._router = _mock_router("SEVERITY: high\nACTION: Rollback immediately")
        store.graph.get_related_decisions = AsyncMock(return_value=[])
        incident = _incident("high")

        classification = await agent._classify_incident(incident, fid)

        assert classification.severity == "high"
        agent._router.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_detects_recurring(self):
        """Past decisions with same title → is_recurring=True."""
        agent, store, fid = _make_agent()
        store.graph.get_related_decisions = AsyncMock(return_value=[
            {"summary": "500 errors in auth at 14:00"},
            {"summary": "500 errors in auth at 15:30"},
        ])
        incident = _incident("medium", "500 errors")

        classification = await agent._classify_incident(incident, fid)

        assert classification.is_recurring is True
        assert classification.recurrence_count == 2
