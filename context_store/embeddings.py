"""
context_store/embeddings.py — Embedding generation.

Generates text embeddings via the OpenAI embeddings API (also used by
Anthropic's recommended embedding approach). Handles chunking for long
documents and batching for efficiency.

The PostgresClient.store_embedding() method accepts pre-computed vectors;
this module is the default path that auto-generates them before storing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from uuid import UUID

import openai

logger = logging.getLogger(__name__)

# Embedding model — 1536 dimensions, matches the pgvector column definition.
# Change here and update the schema if switching models.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# Max tokens per chunk (conservative — model supports 8191)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


@dataclass
class EmbeddingChunk:
    """One chunk of text with its generated vector."""
    chunk_index: int
    content: str
    vector: list[float]


class EmbeddingClient:
    """
    Generates embeddings for artifact text before storing in pgvector.

    Usage:
        embedder = EmbeddingClient()
        chunks = await embedder.embed_text(
            text=prd_content,
            artifact_type="prd",
            artifact_id="docs/prds/feature-123.md",
        )
        for chunk in chunks:
            await pg.store_embedding(
                feature_id=feature_id,
                artifact_type="prd",
                artifact_id="docs/prds/feature-123.md",
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                vector=chunk.vector,
            )
    """

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_EMBED_MODEL):
        self._model = model
        self._api_key = api_key  # resolved lazily on first use
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        """Lazy-init the OpenAI client so ContextStore() can be constructed without credentials."""
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key
                        or os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("ANTHROPIC_API_KEY"),
            )
        return self._client

    async def embed_text(
        self,
        text: str,
        *,
        artifact_type: str,
        artifact_id: str | None = None,
    ) -> list[EmbeddingChunk]:
        """
        Chunk `text`, embed all chunks in one API batch call, return results.
        For short texts (< CHUNK_SIZE words) this is a single embedding call.
        """
        chunks = _split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            return []

        logger.debug(
            "Embedding %d chunk(s) for %s/%s via %s",
            len(chunks), artifact_type, artifact_id, self._model,
        )

        response = await self._get_client().embeddings.create(
            model=self._model,
            input=chunks,
        )

        return [
            EmbeddingChunk(
                chunk_index=i,
                content=chunks[i],
                vector=item.embedding,
            )
            for i, item in enumerate(response.data)
        ]

    async def embed_single(self, text: str) -> list[float]:
        """
        Embed a single short string (e.g. a search query).
        Returns the raw vector — does not store to DB.
        """
        response = await self._get_client().embeddings.create(
            model=self._model,
            input=[text],
        )
        return response.data[0].embedding

    @property
    def model(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping word-count chunks.
    Simple word-boundary split — good enough for document text.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks
