"""
context_store/cache.py — Async Redis client.

Three concerns:
  1. Agent state cache  — short-lived KV for intermediate agent work
  2. Escalation bus     — pub/sub channel for human gate notifications
  3. Distributed lock   — prevent duplicate agent runs on the same feature

Uses redis.asyncio (bundled in the `redis` package ≥ 4.2).
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import UUID

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

logger = logging.getLogger(__name__)

# Escalation bus channel names
CHANNEL_GATES    = "sdlc:gates"       # human gate created / resolved
CHANNEL_ALERTS   = "sdlc:alerts"      # cost spikes, security findings
CHANNEL_INCIDENTS = "sdlc:incidents"  # monitor agent incident events

_LOCK_PREFIX  = "sdlc:lock:"
_STATE_PREFIX = "sdlc:state:"


def _redis_url_from_env() -> str:
    url = os.environ.get("REDIS_URL")
    if url:
        return url
    host     = os.environ.get("REDIS_HOST", "localhost")
    port     = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD", "")
    auth     = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/0"


class CacheClient:
    """
    Async Redis client.

    Usage:
        cache = CacheClient()
        await cache.connect()
        await cache.set_agent_state("intake:abc123", {"step": "interview"}, ttl=3600)
        state = await cache.get_agent_state("intake:abc123")
        await cache.close()

    Or as a context manager:
        async with CacheClient() as cache:
            ...
    """

    def __init__(self, url: str | None = None):
        self._url = url or _redis_url_from_env()
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
        )
        await self._client.ping()
        logger.info("CacheClient connected to Redis")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("CacheClient closed")

    async def __aenter__(self) -> "CacheClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RuntimeError("CacheClient not connected. Call connect() first.")
        return self._client

    # ------------------------------------------------------------------
    # Agent state cache
    # ------------------------------------------------------------------

    async def set_agent_state(self, key: str, value: dict, *, ttl: int = 3600) -> None:
        """Store agent intermediate state. TTL in seconds (default 1 hour)."""
        full_key = f"{_STATE_PREFIX}{key}"
        await self.client.set(full_key, json.dumps(value), ex=ttl)
        logger.debug("State cached: %s  ttl=%ds", full_key, ttl)

    async def get_agent_state(self, key: str) -> dict | None:
        raw = await self.client.get(f"{_STATE_PREFIX}{key}")
        return json.loads(raw) if raw else None

    async def delete_agent_state(self, key: str) -> None:
        await self.client.delete(f"{_STATE_PREFIX}{key}")

    async def extend_agent_state_ttl(self, key: str, ttl: int) -> None:
        await self.client.expire(f"{_STATE_PREFIX}{key}", ttl)

    # ------------------------------------------------------------------
    # Escalation bus (pub/sub)
    # ------------------------------------------------------------------

    async def publish_gate(self, gate_id: UUID, gate_type: str, message: str, feature_id: UUID | None = None) -> None:
        """Publish a human gate event to the escalation bus."""
        payload = {
            "event": "gate_created",
            "gate_id": str(gate_id),
            "gate_type": gate_type,
            "message": message,
            "feature_id": str(feature_id) if feature_id else None,
        }
        await self.client.publish(CHANNEL_GATES, json.dumps(payload))
        logger.info("Published gate event: %s  type=%s", gate_id, gate_type)

    async def publish_alert(self, alert_type: str, payload: dict) -> None:
        """Publish a cost/security alert."""
        envelope = {"event": alert_type, **payload}
        await self.client.publish(CHANNEL_ALERTS, json.dumps(envelope))

    async def publish_incident(self, incident_id: UUID, severity: str, title: str) -> None:
        payload = {
            "event": "incident_created",
            "incident_id": str(incident_id),
            "severity": severity,
            "title": title,
        }
        await self.client.publish(CHANNEL_INCIDENTS, json.dumps(payload))

    @asynccontextmanager
    async def subscribe(self, *channels: str) -> AsyncIterator[PubSub]:
        """
        Async context manager that yields an active PubSub subscription.

        Usage:
            async with cache.subscribe(CHANNEL_GATES, CHANNEL_ALERTS) as pubsub:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        ...
        """
        pubsub: PubSub = self.client.pubsub()
        await pubsub.subscribe(*channels)
        logger.debug("Subscribed to: %s", channels)
        try:
            yield pubsub
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.aclose()

    # ------------------------------------------------------------------
    # Distributed lock
    # Prevents two agent runs from processing the same feature concurrently.
    # ------------------------------------------------------------------

    async def acquire_lock(self, resource: str, *, ttl: int = 300) -> bool:
        """
        Try to acquire a distributed lock for `resource`.
        Returns True if acquired, False if already held.
        TTL in seconds — lock auto-expires if holder crashes.
        """
        key = f"{_LOCK_PREFIX}{resource}"
        acquired = await self.client.set(key, "1", nx=True, ex=ttl)
        if acquired:
            logger.debug("Lock acquired: %s  ttl=%ds", key, ttl)
        else:
            logger.debug("Lock already held: %s", key)
        return bool(acquired)

    async def release_lock(self, resource: str) -> None:
        """Release a previously acquired lock."""
        await self.client.delete(f"{_LOCK_PREFIX}{resource}")
        logger.debug("Lock released: %s", resource)

    @asynccontextmanager
    async def locked(self, resource: str, *, ttl: int = 300) -> AsyncIterator[bool]:
        """
        Context manager for distributed locking.

        Usage:
            async with cache.locked(f"feature:{feature_id}") as acquired:
                if not acquired:
                    return   # another agent is already processing this
                # ... safe to proceed
        """
        acquired = await self.acquire_lock(resource, ttl=ttl)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release_lock(resource)
