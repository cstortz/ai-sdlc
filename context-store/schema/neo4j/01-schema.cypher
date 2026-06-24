// =============================================================================
// AI SDLC — Neo4j Graph Schema
// =============================================================================
// Node Labels:
//   Feature        — top-level work item (synced from Redmine / NL prompt)
//   PRD            — product requirements document
//   ADR            — architecture decision record
//   Implementation — a PR / code change set
//   TestSuite      — collection of tests for an implementation
//   Deployment     — a deployment event (env + version)
//   Incident       — a production incident
//
// Relationship Types:
//   (Feature)        -[:HAS_PRD]->          (PRD)
//   (PRD)            -[:INFORMED_BY]->       (ADR)
//   (ADR)            -[:IMPLEMENTED_BY]->    (Implementation)
//   (PRD)            -[:REALIZED_IN]->       (Implementation)
//   (Implementation) -[:TESTED_BY]->         (TestSuite)
//   (Implementation) -[:DEPLOYED_AS]->       (Deployment)
//   (Deployment)     -[:PRODUCED]->          (Incident)
//   (Incident)       -[:RESOLVED_BY]->       (Implementation)
//   (ADR)            -[:SUPERSEDES]->        (ADR)           // for revised decisions
//   (Feature)        -[:DEPENDS_ON]->        (Feature)       // feature dependencies
//   (Implementation) -[:DEPENDS_ON]->        (Implementation) // code dependencies
// =============================================================================

// ---------------------------------------------------------------------------
// Uniqueness constraints (also create a lookup index automatically)
// ---------------------------------------------------------------------------

CREATE CONSTRAINT feature_id_unique IF NOT EXISTS
    FOR (n:Feature) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT prd_id_unique IF NOT EXISTS
    FOR (n:PRD) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT adr_id_unique IF NOT EXISTS
    FOR (n:ADR) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT implementation_id_unique IF NOT EXISTS
    FOR (n:Implementation) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT testsuite_id_unique IF NOT EXISTS
    FOR (n:TestSuite) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT deployment_id_unique IF NOT EXISTS
    FOR (n:Deployment) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT incident_id_unique IF NOT EXISTS
    FOR (n:Incident) REQUIRE n.id IS UNIQUE;

// ---------------------------------------------------------------------------
// Additional indexes for common query patterns
// ---------------------------------------------------------------------------

// Look up nodes by status (e.g., "all open incidents")
CREATE INDEX feature_status IF NOT EXISTS
    FOR (n:Feature) ON (n.status);

CREATE INDEX deployment_env IF NOT EXISTS
    FOR (n:Deployment) ON (n.environment);

CREATE INDEX incident_severity IF NOT EXISTS
    FOR (n:Incident) ON (n.severity);

CREATE INDEX adr_decision_type IF NOT EXISTS
    FOR (n:ADR) ON (n.decision_type);

// Time-series queries ("what deployed last week?")
CREATE INDEX deployment_created_at IF NOT EXISTS
    FOR (n:Deployment) ON (n.created_at);

CREATE INDEX incident_created_at IF NOT EXISTS
    FOR (n:Incident) ON (n.created_at);

// ---------------------------------------------------------------------------
// Node property schemas (documentation — Neo4j is schemaless but these
// properties are expected by agents and the router)
// ---------------------------------------------------------------------------

// Feature {
//   id: UUID (matches postgres features.id)
//   redmine_id: Integer
//   title: String
//   status: String  // intake | architecture | implementation | testing | review | deploy | done
//   created_at: DateTime
// }

// PRD {
//   id: UUID
//   feature_id: UUID
//   file_path: String  // docs/prds/{feature_id}.md
//   version: Integer
//   status: String     // draft | approved | superseded
//   created_at: DateTime
//   approved_at: DateTime
// }

// ADR {
//   id: UUID
//   feature_id: UUID
//   file_path: String  // docs/adrs/{feature_id}-{adr_id}.md
//   decision_type: String  // data_model | api_design | infra | security | ...
//   status: String     // proposed | accepted | rejected | superseded
//   created_at: DateTime
//   decided_at: DateTime
// }

// Implementation {
//   id: UUID
//   feature_id: UUID
//   github_pr_number: Integer
//   branch: String
//   commit_sha: String
//   status: String  // draft | open | merged | closed
//   created_at: DateTime
//   merged_at: DateTime
// }

// TestSuite {
//   id: UUID
//   implementation_id: UUID
//   test_count: Integer
//   pass_count: Integer
//   fail_count: Integer
//   coverage_pct: Float
//   security_findings: Integer
//   run_at: DateTime
// }

// Deployment {
//   id: UUID
//   implementation_id: UUID
//   environment: String  // staging | production
//   version: String
//   status: String  // pending | running | succeeded | failed | rolled_back
//   triggered_by: String  // agent | human:{name}
//   created_at: DateTime
//   completed_at: DateTime
// }

// Incident {
//   id: UUID
//   deployment_id: UUID
//   severity: String  // low | medium | high | critical
//   title: String
//   description: String
//   status: String  // open | investigating | resolved
//   created_at: DateTime
//   resolved_at: DateTime
// }

// ---------------------------------------------------------------------------
// Example traversal queries (for reference — run in Neo4j Browser or via Bolt)
// ---------------------------------------------------------------------------

// Full lineage for a feature:
// MATCH path = (f:Feature {id: $id})-[:HAS_PRD]->(p:PRD)
//              -[:INFORMED_BY]->(a:ADR)
//              -[:IMPLEMENTED_BY]->(i:Implementation)
//              -[:TESTED_BY]->(t:TestSuite)
//              -[:DEPLOYED_AS]->(d:Deployment)
// RETURN path

// All incidents linked back to their originating feature:
// MATCH (f:Feature)-[:HAS_PRD]->(:PRD)-[:REALIZED_IN]->(:Implementation)
//       -[:DEPLOYED_AS]->(d:Deployment)-[:PRODUCED]->(inc:Incident)
// WHERE inc.status = 'open'
// RETURN f.title, inc.severity, inc.title, d.environment
// ORDER BY inc.severity DESC

// Recurring incidents (same feature, 3+ incidents):
// MATCH (f:Feature)-[:HAS_PRD]->(:PRD)-[:REALIZED_IN]->(:Implementation)
//       -[:DEPLOYED_AS]->(:Deployment)-[:PRODUCED]->(inc:Incident)
// WITH f, count(inc) AS incident_count
// WHERE incident_count >= 3
// RETURN f.title, incident_count
// ORDER BY incident_count DESC
