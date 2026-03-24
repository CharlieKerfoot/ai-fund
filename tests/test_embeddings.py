"""Tests for EmbeddingStore."""

from __future__ import annotations

import duckdb
import pytest

from ingest.extractors.embedding import EmbeddingStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_store():
    """In-memory DuckDB-backed EmbeddingStore."""
    conn = duckdb.connect(":memory:")
    store = EmbeddingStore(conn, dimension=64)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_and_store(embedding_store: EmbeddingStore):
    """Embed text, store it, verify retrieval."""
    store = embedding_store

    doc_id = "doc_001"
    text = "Apple reported strong quarterly earnings with revenue growth."
    metadata = {"ticker": "AAPL", "source": "earnings"}

    store.store_embedding(doc_id, text, metadata)

    # Verify stored in DB
    row = store._conn.execute(
        "SELECT doc_id, metadata FROM embeddings WHERE doc_id = ?",
        [doc_id],
    ).fetchone()

    assert row is not None
    assert row[0] == doc_id


def test_embed_returns_correct_dimension(embedding_store: EmbeddingStore):
    """Verify embedding dimension matches configured dimension."""
    import numpy as np

    store = embedding_store
    vec = store.embed_text("Test text for dimension check.")
    assert vec.shape == (64,)
    # Should be normalized (unit vector)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5 or norm == pytest.approx(1.0, abs=1e-5)


def test_find_similar(embedding_store: EmbeddingStore):
    """Store 3 docs, query for similar, verify ranking."""
    store = embedding_store

    docs = [
        ("doc_tech_1", "Apple iPhone revenue growth technology smartphone", {"category": "tech"}),
        ("doc_tech_2", "Microsoft Azure cloud computing revenue profit technology", {"category": "tech"}),
        ("doc_energy_1", "Oil drilling offshore petroleum energy extraction fossil fuel", {"category": "energy"}),
    ]

    for doc_id, text, meta in docs:
        store.store_embedding(doc_id, text, meta)

    # Query for tech-related content — tech docs should rank higher than energy
    results = store.find_similar("Apple Microsoft technology revenue", top_k=3)

    assert len(results) > 0
    assert len(results) <= 3

    # Each result should have required keys
    for r in results:
        assert "doc_id" in r
        assert "score" in r
        assert "metadata" in r

    # Top result should be one of the tech docs (not energy)
    top_ids = [r["doc_id"] for r in results[:2]]
    assert "doc_energy_1" not in top_ids[:1] or len(results) == 1


def test_find_similar_empty_store():
    """find_similar on empty store returns empty list."""
    conn = duckdb.connect(":memory:")
    store = EmbeddingStore(conn, dimension=32)
    results = store.find_similar("some query", top_k=5)
    assert results == []


def test_store_and_retrieve_multiple(embedding_store: EmbeddingStore):
    """Store multiple embeddings and verify all are retrievable."""
    store = embedding_store

    texts = [
        ("id1", "Revenue growth exceeded expectations this quarter"),
        ("id2", "Risk factors include supply chain disruptions"),
        ("id3", "Management confident about future guidance"),
        ("id4", "Going concern language detected in the filing"),
        ("id5", "Strong balance sheet with minimal debt"),
    ]

    for doc_id, text in texts:
        store.store_embedding(doc_id, text, {"idx": doc_id})

    count = store._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert count == len(texts)

    results = store.find_similar("financial outlook guidance", top_k=3)
    assert len(results) == 3
