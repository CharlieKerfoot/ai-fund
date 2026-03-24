"""Text embedding store using TF-IDF + SVD and DuckDB storage."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Store and retrieve text embeddings using TF-IDF + TruncatedSVD."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS embeddings (
        doc_id VARCHAR PRIMARY KEY,
        metadata JSON,
        embedding BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self, conn: Any, dimension: int = 384) -> None:
        self._conn = conn
        self._dimension = dimension
        self._conn.execute(self._DDL)

        # Lazy-initialized sklearn components
        self._vectorizer = None
        self._svd = None
        self._is_fitted = False

        # In-memory corpus for re-fitting the vectorizer
        self._corpus: list[str] = []
        self._corpus_ids: list[str] = []

    def _ensure_fitted(self, new_text: str) -> None:
        """Fit or re-fit the vectorizer on the accumulated corpus."""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._corpus.append(new_text)

        n_components = min(self._dimension, len(self._corpus) - 1)
        if n_components < 1:
            n_components = 1

        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
        )
        tfidf_matrix = self._vectorizer.fit_transform(self._corpus)

        actual_components = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if actual_components < 1:
            actual_components = 1

        self._svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self._svd.fit(tfidf_matrix)
        self._is_fitted = True

    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector using TF-IDF + SVD fallback."""
        self._ensure_fitted(text)
        tfidf = self._vectorizer.transform([text])
        vec = self._svd.transform(tfidf)[0]

        # Pad or truncate to the target dimension
        if len(vec) < self._dimension:
            vec = np.pad(vec, (0, self._dimension - len(vec)))
        else:
            vec = vec[: self._dimension]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def store_embedding(self, doc_id: str, text: str, metadata: dict) -> None:
        """Compute and store embedding in DuckDB."""
        vec = self.embed_text(text)
        embedding_blob = pickle.dumps(vec)
        metadata_json = json.dumps(metadata)
        created_at = datetime.now(tz=UTC)

        self._conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (doc_id, metadata, embedding, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [doc_id, metadata_json, embedding_blob, created_at],
        )
        self._corpus_ids.append(doc_id)
        logger.debug("Stored embedding for doc_id=%s", doc_id)

    def find_similar(self, query: str, top_k: int = 5) -> list[dict]:
        """Find similar documents by cosine similarity."""
        query_vec = self.embed_text(query)

        rows = self._conn.execute(
            "SELECT doc_id, metadata, embedding FROM embeddings"
        ).fetchall()

        if not rows:
            return []

        scores: list[tuple[float, str, dict]] = []
        for doc_id, metadata_json, embedding_blob in rows:
            try:
                doc_vec = pickle.loads(embedding_blob)  # noqa: S301
                # Vectors are already normalized; dot product == cosine similarity
                score = float(np.dot(query_vec, doc_vec))
                metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                scores.append((score, doc_id, metadata))
            except Exception as exc:
                logger.warning("Error computing similarity for doc_id=%s: %s", doc_id, exc)

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {"doc_id": doc_id, "score": score, "metadata": meta}
            for score, doc_id, meta in scores[:top_k]
        ]
