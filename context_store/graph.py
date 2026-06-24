"""
context_store/graph.py — Async Neo4j client.

Provides typed methods for every node label and relationship type in the
SDLC traceability graph:

  Nodes:        Feature, PRD, ADR, Implementation, TestSuite, Deployment, Incident
  Relationships: HAS_PRD, INFORMED_BY, IMPLEMENTED_BY, REALIZED_IN,
                 TESTED_BY, DEPLOYED_AS, PRODUCED, RESOLVED_BY,
                 SUPERSEDES, DEPENDS_ON

Uses the official neo4j Python async driver (neo4j.AsyncGraphDatabase).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import neo4j
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


def _bolt_uri_from_env() -> str:
    return os.environ.get(
        "NEO4J_URI",
        f"bolt://{os.environ.get('NEO4J_HOST', 'localhost')}:7687",
    )


class GraphClient:
    """
    Async Neo4j client.

    Usage:
        graph = GraphClient()
        await graph.connect()
        node_id = await graph.create_feature(id=feature_id, title="Login feature")
        await graph.close()

    Or as a context manager:
        async with GraphClient() as graph:
            ...
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._uri = uri or _bolt_uri_from_env()
        self._user = user or os.environ.get("NEO4J_USER", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "")
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        await self._driver.verify_connectivity()
        logger.info("GraphClient connected to %s", self._uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            logger.info("GraphClient closed")

    async def __aenter__(self) -> "GraphClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    @property
    def driver(self) -> AsyncDriver:
        if not self._driver:
            raise RuntimeError("GraphClient not connected. Call connect() first.")
        return self._driver

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    async def _run(self, cypher: str, **params) -> list[dict]:
        """Execute a Cypher query and return results as a list of dicts."""
        async with self._driver.session() as session:
            result = await session.run(cypher, **params)
            return [dict(record) async for record in result]

    async def _run_write(self, cypher: str, **params) -> None:
        async with self._driver.session() as session:
            await session.run(cypher, **params)

    # ------------------------------------------------------------------
    # Node creation — one method per label
    # ------------------------------------------------------------------

    async def create_feature(self, *, id: UUID, title: str, redmine_id: int | None = None, status: str = "intake") -> None:
        await self._run_write(
            """
            MERGE (n:Feature {id: $id})
            SET n.title = $title,
                n.redmine_id = $redmine_id,
                n.status = $status,
                n.created_at = $now
            """,
            id=str(id), title=title, redmine_id=redmine_id,
            status=status, now=_now(),
        )

    async def create_prd(self, *, id: UUID, feature_id: UUID, file_path: str, version: int = 1, status: str = "draft") -> None:
        await self._run_write(
            """
            MERGE (n:PRD {id: $id})
            SET n.feature_id = $feature_id,
                n.file_path = $file_path,
                n.version = $version,
                n.status = $status,
                n.created_at = $now
            WITH n
            MATCH (f:Feature {id: $feature_id})
            MERGE (f)-[:HAS_PRD]->(n)
            """,
            id=str(id), feature_id=str(feature_id),
            file_path=file_path, version=version, status=status, now=_now(),
        )

    async def create_adr(self, *, id: UUID, feature_id: UUID, file_path: str, decision_type: str, status: str = "proposed") -> None:
        await self._run_write(
            """
            MERGE (n:ADR {id: $id})
            SET n.feature_id = $feature_id,
                n.file_path = $file_path,
                n.decision_type = $decision_type,
                n.status = $status,
                n.created_at = $now
            WITH n
            MATCH (p:PRD {feature_id: $feature_id})
            MERGE (p)-[:INFORMED_BY]->(n)
            """,
            id=str(id), feature_id=str(feature_id),
            file_path=file_path, decision_type=decision_type,
            status=status, now=_now(),
        )

    async def create_implementation(
        self, *, id: UUID, feature_id: UUID,
        github_pr_number: int | None = None,
        branch: str | None = None,
        status: str = "draft",
    ) -> None:
        await self._run_write(
            """
            MERGE (n:Implementation {id: $id})
            SET n.feature_id = $feature_id,
                n.github_pr_number = $pr_number,
                n.branch = $branch,
                n.status = $status,
                n.created_at = $now
            WITH n
            MATCH (a:ADR {feature_id: $feature_id})
            MERGE (a)-[:IMPLEMENTED_BY]->(n)
            WITH n
            MATCH (p:PRD {feature_id: $feature_id})
            MERGE (p)-[:REALIZED_IN]->(n)
            """,
            id=str(id), feature_id=str(feature_id),
            pr_number=github_pr_number, branch=branch,
            status=status, now=_now(),
        )

    async def create_test_suite(
        self, *, id: UUID, implementation_id: UUID,
        test_count: int = 0, pass_count: int = 0,
        fail_count: int = 0, coverage_pct: float = 0.0,
        security_findings: int = 0,
    ) -> None:
        await self._run_write(
            """
            MERGE (n:TestSuite {id: $id})
            SET n.implementation_id = $impl_id,
                n.test_count = $test_count,
                n.pass_count = $pass_count,
                n.fail_count = $fail_count,
                n.coverage_pct = $coverage_pct,
                n.security_findings = $security_findings,
                n.run_at = $now
            WITH n
            MATCH (i:Implementation {id: $impl_id})
            MERGE (i)-[:TESTED_BY]->(n)
            """,
            id=str(id), impl_id=str(implementation_id),
            test_count=test_count, pass_count=pass_count,
            fail_count=fail_count, coverage_pct=coverage_pct,
            security_findings=security_findings, now=_now(),
        )

    async def create_deployment(
        self, *, id: UUID, implementation_id: UUID,
        environment: str, version: str,
        triggered_by: str = "agent",
        status: str = "pending",
    ) -> None:
        await self._run_write(
            """
            MERGE (n:Deployment {id: $id})
            SET n.implementation_id = $impl_id,
                n.environment = $environment,
                n.version = $version,
                n.triggered_by = $triggered_by,
                n.status = $status,
                n.created_at = $now
            WITH n
            MATCH (i:Implementation {id: $impl_id})
            MERGE (i)-[:DEPLOYED_AS]->(n)
            """,
            id=str(id), impl_id=str(implementation_id),
            environment=environment, version=version,
            triggered_by=triggered_by, status=status, now=_now(),
        )

    async def create_incident(
        self, *, id: UUID, deployment_id: UUID,
        severity: str, title: str,
        description: str | None = None,
        status: str = "open",
    ) -> None:
        await self._run_write(
            """
            MERGE (n:Incident {id: $id})
            SET n.deployment_id = $dep_id,
                n.severity = $severity,
                n.title = $title,
                n.description = $description,
                n.status = $status,
                n.created_at = $now
            WITH n
            MATCH (d:Deployment {id: $dep_id})
            MERGE (d)-[:PRODUCED]->(n)
            """,
            id=str(id), dep_id=str(deployment_id),
            severity=severity, title=title,
            description=description, status=status, now=_now(),
        )

    # ------------------------------------------------------------------
    # Relationship helpers
    # ------------------------------------------------------------------

    async def link_incident_resolved_by(self, incident_id: UUID, implementation_id: UUID) -> None:
        """Mark that a fix implementation resolves an incident."""
        await self._run_write(
            """
            MATCH (inc:Incident {id: $inc_id})
            MATCH (impl:Implementation {id: $impl_id})
            MERGE (inc)-[:RESOLVED_BY]->(impl)
            """,
            inc_id=str(incident_id), impl_id=str(implementation_id),
        )

    async def supersede_adr(self, old_adr_id: UUID, new_adr_id: UUID) -> None:
        """Mark that a new ADR supersedes an old one."""
        await self._run_write(
            """
            MATCH (new:ADR {id: $new_id})
            MATCH (old:ADR {id: $old_id})
            MERGE (new)-[:SUPERSEDES]->(old)
            SET old.status = 'superseded'
            """,
            new_id=str(new_adr_id), old_id=str(old_adr_id),
        )

    async def update_node_status(self, label: str, node_id: UUID, status: str) -> None:
        await self._run_write(
            f"MATCH (n:{label} {{id: $id}}) SET n.status = $status",
            id=str(node_id), status=status,
        )

    # ------------------------------------------------------------------
    # Traversal queries
    # ------------------------------------------------------------------

    async def get_feature_lineage(self, feature_id: UUID) -> list[dict]:
        """
        Return the full artifact chain for a feature:
        Feature → PRD → ADR → Implementation → TestSuite → Deployment → Incident
        """
        rows = await self._run(
            """
            MATCH (f:Feature {id: $fid})
            OPTIONAL MATCH (f)-[:HAS_PRD]->(p:PRD)
            OPTIONAL MATCH (p)-[:INFORMED_BY]->(a:ADR)
            OPTIONAL MATCH (a)-[:IMPLEMENTED_BY]->(i:Implementation)
            OPTIONAL MATCH (i)-[:TESTED_BY]->(t:TestSuite)
            OPTIONAL MATCH (i)-[:DEPLOYED_AS]->(d:Deployment)
            OPTIONAL MATCH (d)-[:PRODUCED]->(inc:Incident)
            RETURN f, p, a, i, t, d, inc
            """,
            fid=str(feature_id),
        )
        return rows

    async def get_open_incidents(self) -> list[dict]:
        """All open incidents with their parent feature context."""
        return await self._run(
            """
            MATCH (f:Feature)-[:HAS_PRD]->(:PRD)-[:REALIZED_IN]->(:Implementation)
                  -[:DEPLOYED_AS]->(d:Deployment)-[:PRODUCED]->(inc:Incident)
            WHERE inc.status = 'open'
            RETURN f.id AS feature_id, f.title AS feature_title,
                   inc.id AS incident_id, inc.severity AS severity,
                   inc.title AS incident_title, d.environment AS environment,
                   inc.created_at AS created_at
            ORDER BY
              CASE inc.severity
                WHEN 'critical' THEN 0
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                ELSE 3
              END, inc.created_at
            """
        )

    async def get_recurring_incidents(self, threshold: int = 3) -> list[dict]:
        """Features that have triggered >= threshold incidents (signals architectural issues)."""
        return await self._run(
            """
            MATCH (f:Feature)-[:HAS_PRD]->(:PRD)-[:REALIZED_IN]->(:Implementation)
                  -[:DEPLOYED_AS]->(:Deployment)-[:PRODUCED]->(inc:Incident)
            WITH f, count(inc) AS incident_count
            WHERE incident_count >= $threshold
            RETURN f.id AS feature_id, f.title AS feature_title,
                   incident_count
            ORDER BY incident_count DESC
            """,
            threshold=threshold,
        )

    async def get_related_decisions(self, feature_id: UUID) -> list[dict]:
        """All ADRs for a feature — useful for feeding context into agents."""
        return await self._run(
            """
            MATCH (f:Feature {id: $fid})-[:HAS_PRD]->(:PRD)-[:INFORMED_BY]->(a:ADR)
            RETURN a.id AS adr_id, a.file_path AS file_path,
                   a.decision_type AS decision_type, a.status AS status
            ORDER BY a.created_at
            """,
            fid=str(feature_id),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
