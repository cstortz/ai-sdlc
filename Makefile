# AI SDLC — Developer shortcuts
# Requires: docker compose v2, Python 3.11+

.PHONY: help up down restart logs shell \
        test test-agents test-router test-store test-watch \
        lint fmt migrate migrate-down \
        pipeline scan-all

# ─── Default ─────────────────────────────────────────────────────────────────

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Infrastructure ───────────────────────────────────────────────────────────

up: ## Start all services (Postgres, Redis, Neo4j)
	docker compose up -d
	@echo "Waiting for Postgres..."
	@until docker compose exec -T postgres pg_isready -U sdlc 2>/dev/null; do sleep 1; done
	@echo "✓ All services ready"

down: ## Stop all services
	docker compose down

restart: down up ## Restart all services

logs: ## Tail logs from all services
	docker compose logs -f

shell: ## Open a shell in the app container (if running)
	docker compose exec app bash

# ─── Database ─────────────────────────────────────────────────────────────────

migrate: ## Run Alembic migrations (up)
	python -m alembic upgrade head

migrate-down: ## Roll back last migration
	python -m alembic downgrade -1

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add users table")
	python -m alembic revision --autogenerate -m "$(MSG)"

# ─── Tests ────────────────────────────────────────────────────────────────────

test: ## Run all unit tests
	python -m pytest agents/ router/tests/ context_store/tests/ \
		-v --asyncio-mode=auto --tb=short -p no:cacheprovider

test-agents: ## Run agent tests only (L1–L7)
	python -m pytest agents/ -v --asyncio-mode=auto --tb=short

test-router: ## Run model router tests
	python -m pytest router/tests/ -v --asyncio-mode=auto --tb=short

test-store: ## Run context store tests
	python -m pytest context_store/tests/ -v --asyncio-mode=auto --tb=short

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw -- -v --asyncio-mode=auto --tb=short

test-layer: ## Run a single layer's tests (usage: make test-layer L=intake)
	python -m pytest agents/$(L)/tests/ -v --asyncio-mode=auto --tb=short

# ─── Lint / Format ────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	ruff check . --select E,W,F,I --ignore E501

fmt: ## Auto-format with ruff
	ruff format .
	ruff check . --select I --fix

# ─── Pipeline ─────────────────────────────────────────────────────────────────

pipeline: ## Run the full pipeline (usage: make pipeline P="Add user login")
	python -m agents.pipeline --prompt "$(P)"

pipeline-resume: ## Resume pipeline from feature UUID (usage: make pipeline-resume F=<uuid>)
	python -m agents.pipeline --resume $(F)

approve: ## Approve a human gate (usage: make approve G=<gate-uuid>)
	python -m agents.pipeline --approve-gate $(G)

reject: ## Reject a human gate (usage: make reject G=<gate-uuid>)
	python -m agents.pipeline --reject-gate $(G)

scan-all: ## Triage all open incidents (Monitor Agent cron mode)
	python -m agents.monitor.agent --scan-all
