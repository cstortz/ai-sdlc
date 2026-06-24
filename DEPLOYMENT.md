# AI SDLC — Infrastructure Deployment Guide

This document covers the full lifecycle of the infrastructure stack:
dev environment (Docker Compose) and production (Helm on Kubernetes).

---

## Table of Contents

1. [Architecture Summary](#1-architecture-summary)
2. [Prerequisites](#2-prerequisites)
3. [Dev Environment — Docker Compose](#3-dev-environment--docker-compose)
4. [Production — Helm on Kubernetes](#4-production--helm-on-kubernetes)
5. [Secrets Management](#5-secrets-management)
6. [Post-Deploy Verification](#6-post-deploy-verification)
7. [Upgrading](#7-upgrading)
8. [Teardown](#8-teardown)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture Summary

| Service | Purpose | Dev port | Prod access |
|---|---|---|---|
| **Postgres 16 + pgvector** | Agent state, audit trail, embeddings | 5432 | ClusterIP `sdlc-postgres-postgresql:5432` |
| **Neo4j 5** | Traceability graph | 7474 (UI), 7687 (Bolt) | ClusterIP, port-forward to access |
| **Redis 7** | Agent state cache, escalation bus | 6379 | ClusterIP `sdlc-redis-master:6379` |
| **Redmine 5** | Work item / project management | 3000 | ClusterIP, expose via ingress |
| **Adminer** | Postgres UI (dev only) | 8080 | Not deployed in prod |

Storage: all persistent volumes use the `nfs-client` storage class
(default, `nfs-subdir-external-provisioner`). PVs are provisioned automatically.

---

## 2. Prerequisites

### Dev

| Tool | Min version | Install |
|---|---|---|
| Docker | 24+ | https://docs.docker.com/get-docker/ |
| Docker Compose | v2 (`docker compose`) | Bundled with Docker Desktop; `apt install docker-compose-plugin` on Linux |

### Production

| Tool | Min version | Install |
|---|---|---|
| kubectl | 1.28+ | https://kubernetes.io/docs/tasks/tools/ |
| Helm | 3.14+ | https://helm.sh/docs/intro/install/ |
| Access to the cluster | — | `~/.kube/config` pointed at your k3s/kubeadm cluster |
| `nfs-client` storage class | — | Must exist: `kubectl get storageclass` |

Verify:
```bash
kubectl version --short
helm version
kubectl get storageclass    # should show nfs-client (default)
```

---

## 3. Dev Environment — Docker Compose

### First-time setup

```bash
cd ai-sdlc

# 1. Create your local .env from the template
cp .env.example .env
# Edit .env — change every CHANGE_ME value before proceeding
vim .env

# 2. Start all services
docker compose up -d

# 3. Tail logs to confirm healthy startup (takes ~30s for Redmine)
docker compose logs -f
```

### Service URLs (after startup)

| Service | URL |
|---|---|
| Redmine | http://localhost:3000 (admin / check .env for password) |
| Neo4j Browser | http://localhost:7474 (neo4j / NEO4J_PASSWORD from .env) |
| Adminer (DB UI) | Start with: `docker compose --profile dev up -d adminer` then http://localhost:8080 |
| Postgres | localhost:5432 (connect with psql or Adminer) |
| Redis | localhost:6379 |

### Run the Neo4j schema

After the Neo4j container is healthy:

```bash
# Copy the schema file into the container and execute it
docker compose exec neo4j cypher-shell \
  -u neo4j -p "$(grep NEO4J_PASSWORD .env | cut -d= -f2)" \
  -f /var/lib/neo4j/import/01-schema.cypher
```

### Verify Postgres schema

```bash
docker compose exec postgres \
  psql -U sdlc -d sdlc -c "\dt"
# Should list: features, agent_runs, decisions, human_gates, embeddings, context_snapshots
```

### Verify pgvector

```bash
docker compose exec postgres \
  psql -U sdlc -d sdlc -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

### Useful day-to-day commands

```bash
docker compose ps                      # service status
docker compose restart redmine         # restart a single service
docker compose logs -f postgres        # stream logs from one service
docker compose down                    # stop all (data persists in volumes)
docker compose down -v                 # stop AND delete all data ⚠
```

---

## 4. Production — Helm on Kubernetes

### 4.1 Prepare secrets

```bash
cd ai-sdlc

# Copy the secrets template
cp helm/secrets.yaml helm/secrets.yaml.bak   # optional: keep a blank template

# Edit secrets — fill in every CHANGE_ME value
vim helm/secrets.yaml

# Encrypt the file with vim's built-in blowfish2 encryption
# You will be prompted for a password — store it somewhere safe (password manager)
vi -x helm/secrets.yaml
# Vim writes the encrypted file in-place.
# To decrypt and re-edit later: vi -x helm/secrets.yaml (same command)
```

> **Never commit `helm/secrets.yaml` in plaintext.**
> It is listed in both `.gitignore` and `helm/.helmignore`.

### 4.2 Create the namespace

```bash
kubectl create namespace ai-sdlc
```

### 4.3 Fetch Helm dependencies

Run this once, and again whenever `helm/Chart.yaml` dependencies change:

```bash
helm dependency update helm/
# Downloads Bitnami postgresql, redis, redmine charts + Neo4j chart
# into helm/charts/
```

### 4.4 Install

```bash
# Decrypt secrets first (vim opens the file; save+quit to decrypt to stdout below)
# OR pipe decrypted content directly:

helm install ai-sdlc helm/ \
  --namespace ai-sdlc \
  -f helm/values.yaml \
  -f helm/secrets.yaml \
  --set-file agentConfig.modelsConfig=config/models.yaml \
  --set-file agentConfig.agentsConfig=config/agents.yaml \
  --timeout 10m \
  --wait
```

> `--wait` blocks until all pods are ready or the timeout is reached.
> `--timeout 10m` gives Redmine time to run its database migrations on first start.

### 4.5 Run the Neo4j schema (first install only)

```bash
# Get the Neo4j pod name
NEO4J_POD=$(kubectl get pod -n ai-sdlc -l app.kubernetes.io/name=neo4j -o jsonpath='{.items[0].metadata.name}')

# Copy the schema file into the pod
kubectl cp context-store/schema/neo4j/01-schema.cypher \
  ai-sdlc/$NEO4J_POD:/tmp/01-schema.cypher

# Execute it
NEO4J_PASS=$(kubectl get secret -n ai-sdlc ai-sdlc-neo4j -o jsonpath='{.data.NEO4J_AUTH}' | base64 -d | cut -d/ -f2)
kubectl exec -n ai-sdlc $NEO4J_POD -- \
  cypher-shell -u neo4j -p "$NEO4J_PASS" -f /tmp/01-schema.cypher
```

### 4.6 First-time Redmine setup

```bash
# Port-forward Redmine to get your API key
kubectl port-forward svc/ai-sdlc-redmine -n ai-sdlc 3000:3000 &

# Open http://localhost:3000
# Log in as admin with the password from secrets.yaml → redmine.redminePassword
# Go to: My Account → API access key → Show → copy the key
# Add it to secrets.yaml (decrypt first with vi -x), then re-encrypt and upgrade:

helm upgrade ai-sdlc helm/ \
  --namespace ai-sdlc \
  -f helm/values.yaml \
  -f helm/secrets.yaml \
  --set-file agentConfig.modelsConfig=config/models.yaml \
  --set-file agentConfig.agentsConfig=config/agents.yaml
```

---

## 5. Secrets Management

### File: `helm/secrets.yaml`

Contains all passwords and API keys. Two-state lifecycle:

| State | How to get there |
|---|---|
| **Encrypted** (safe to store) | `vi -x helm/secrets.yaml` → save → quit |
| **Decrypted** (edit mode) | `vi -x helm/secrets.yaml` → vim prompts for password → edit → save |

The file uses vim's `blowfish2` encryption. This is a pragmatic local secret store — suitable for a solo operator on a private cluster.

### API keys loaded into the cluster

The `modelApiKeys` block in `secrets.yaml` is rendered by Helm into a Kubernetes Secret, then mounted as environment variables in agent pods:

| Env var | Provider |
|---|---|
| `ANTHROPIC_API_KEY` | Claude (primary) |
| `GROQ_API_KEY` | Groq (open-source fallback) |
| `TOGETHER_API_KEY` | Together AI (open-source fallback) |

### Rotating a secret

1. `vi -x helm/secrets.yaml` (decrypt, edit the value, save, re-encrypts)
2. `helm upgrade ai-sdlc helm/ -f helm/values.yaml -f helm/secrets.yaml ...`
3. Pods with the changed secret will rolling-restart automatically.

---

## 6. Post-Deploy Verification

Run these checks after every install or upgrade.

### Check all pods are running

```bash
kubectl get pods -n ai-sdlc
# Expected: all pods in Running state, no CrashLoopBackOff
```

### Check PVCs are bound

```bash
kubectl get pvc -n ai-sdlc
# All PVCs should show STATUS: Bound
# STORAGECLASS: nfs-client
```

### Verify Postgres schema and pgvector

```bash
PG_POD=$(kubectl get pod -n ai-sdlc -l app.kubernetes.io/name=postgresql -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ai-sdlc $PG_POD -- psql -U sdlc -d sdlc -c "\dt"
kubectl exec -n ai-sdlc $PG_POD -- psql -U sdlc -d sdlc \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

### Verify Neo4j constraints

```bash
NEO4J_POD=$(kubectl get pod -n ai-sdlc -l app.kubernetes.io/name=neo4j -o jsonpath='{.items[0].metadata.name}')
NEO4J_PASS=$(kubectl get secret -n ai-sdlc ai-sdlc-neo4j -o jsonpath='{.data.NEO4J_AUTH}' | base64 -d | cut -d/ -f2)
kubectl exec -n ai-sdlc $NEO4J_POD -- \
  cypher-shell -u neo4j -p "$NEO4J_PASS" "SHOW CONSTRAINTS;"
# Should list 7 uniqueness constraints (one per node label)
```

### Verify Redis

```bash
REDIS_POD=$(kubectl get pod -n ai-sdlc -l app.kubernetes.io/name=redis -o jsonpath='{.items[0].metadata.name}')
REDIS_PASS=$(kubectl get secret -n ai-sdlc ai-sdlc-redis -o jsonpath='{.data.redis-password}' | base64 -d)
kubectl exec -n ai-sdlc $REDIS_POD -- redis-cli -a "$REDIS_PASS" ping
# Expected: PONG
```

### Quick port-forward for browser access

```bash
# Neo4j Browser
kubectl port-forward svc/ai-sdlc-neo4j-neo4j -n ai-sdlc 7474:7474 7687:7687 &

# Redmine
kubectl port-forward svc/ai-sdlc-redmine -n ai-sdlc 3000:3000 &

# Postgres (via psql or Adminer locally)
kubectl port-forward svc/ai-sdlc-postgres-postgresql -n ai-sdlc 5432:5432 &
```

---

## 7. Upgrading

### Update agent config only (no infra change)

```bash
helm upgrade ai-sdlc helm/ \
  --namespace ai-sdlc \
  -f helm/values.yaml \
  -f helm/secrets.yaml \
  --set-file agentConfig.modelsConfig=config/models.yaml \
  --set-file agentConfig.agentsConfig=config/agents.yaml \
  --reuse-values
```

### Update Helm chart dependencies

```bash
helm dependency update helm/
helm upgrade ai-sdlc helm/ \
  --namespace ai-sdlc \
  -f helm/values.yaml \
  -f helm/secrets.yaml \
  --set-file agentConfig.modelsConfig=config/models.yaml \
  --set-file agentConfig.agentsConfig=config/agents.yaml \
  --timeout 10m --wait
```

### Roll back to previous release

```bash
helm history ai-sdlc -n ai-sdlc          # list releases
helm rollback ai-sdlc <REVISION> -n ai-sdlc
```

### Resize a PVC (nfs-client supports expansion)

```bash
kubectl patch pvc <pvc-name> -n ai-sdlc \
  -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
```

---

## 8. Teardown

### Dev — stop without data loss

```bash
docker compose down
# Volumes are preserved. Data survives.
```

### Dev — full reset (deletes all data)

```bash
docker compose down -v
```

### Production — uninstall (preserves PVCs by default)

```bash
helm uninstall ai-sdlc -n ai-sdlc
# Helm removes Deployments, Services, ConfigMaps, Secrets.
# PVCs are NOT deleted by default — data is safe.
```

> **Warning:** The `nfs-client` provisioner has `ReclaimPolicy: Delete`.
> If you manually delete a PVC, the underlying NFS share and all data
> in it will be permanently removed.

### Production — full teardown including data

```bash
helm uninstall ai-sdlc -n ai-sdlc
kubectl delete pvc --all -n ai-sdlc    # ⚠ deletes all PVs and NFS data
kubectl delete namespace ai-sdlc
```

---

## 9. Troubleshooting

### Redmine fails to start — database connection error

Redmine starts before Postgres is fully ready on first boot.

```bash
kubectl rollout restart deployment/ai-sdlc-redmine -n ai-sdlc
```

### Neo4j pod stuck in `Pending`

Check PVC binding:

```bash
kubectl describe pvc -n ai-sdlc | grep -A5 "Events"
```

If the NFS provisioner isn't creating the PV, verify the provisioner pod is running:

```bash
kubectl get pods -n nfs-provisioner   # or whichever namespace it's in
```

### pgvector extension missing after Postgres starts

The init scripts only run on a **fresh** volume. If you're seeing a pre-existing volume without pgvector:

```bash
PG_POD=$(kubectl get pod -n ai-sdlc -l app.kubernetes.io/name=postgresql -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ai-sdlc $PG_POD -- psql -U sdlc -d sdlc \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Helm upgrade fails: "cannot patch immutable field"

Some Kubernetes resources (e.g., Service `clusterIP`) are immutable after creation. Force-upgrade:

```bash
helm upgrade ai-sdlc helm/ --namespace ai-sdlc \
  -f helm/values.yaml -f helm/secrets.yaml \
  --force
```

### Check what Helm would change before applying

```bash
helm diff upgrade ai-sdlc helm/ \
  --namespace ai-sdlc \
  -f helm/values.yaml \
  -f helm/secrets.yaml
# Requires: helm plugin install https://github.com/databus23/helm-diff
```

### Decrypt secrets if you forget the vim encryption password

There is no recovery path — `blowfish2` encryption is one-way without the password.
Always store the encryption password in a password manager alongside the other credentials.
