"""Ranking features.

This module is deliberately the single definition of how a (query, chunk) pair
becomes a feature vector. Training and serving both import it, which is the
only reliable way to avoid train/serve skew: if the feature order or scaling
ever drifts between the two paths, a learned ranker silently degrades to noise
and nothing in the test suite notices.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from secrag.core.types import ChunkKind, FilingSection, ScoredChunk
from secrag.retrieval.bm25 import tokenize

# Order is part of the model contract. Append only, never reorder.
FEATURE_NAMES: tuple[str, ...] = (
    "rrf_score",
    "dense_score",
    "dense_rank_inv",
    "bm25_score",
    "bm25_rank_inv",
    "splade_score",
    "splade_rank_inv",
    "arm_count",
    "token_estimate",
    "is_table",
    "section_id",
    "query_tokens",
    "lexical_overlap",
    "lexical_coverage",
    "numeric_density",
    "year_recency",
)

_SECTION_IDS = {section: i for i, section in enumerate(FilingSection)}
_CURRENT_YEAR = 2025


def _rank_inv(components: dict[str, float], arm: str) -> float:
    """Reciprocal rank for one arm, or 0 when that arm did not retrieve it.

    Reciprocal rather than raw rank so the feature is bounded and so the gap
    between positions 1 and 2 counts for more than the gap between 30 and 31,
    which is how relevance actually behaves.
    """
    rank = components.get(f"{arm}_rank")
    return 1.0 / rank if rank else 0.0


def extract_features(query: str, candidates: Sequence[ScoredChunk]) -> np.ndarray:
    """Build the (n_candidates, n_features) matrix for one query."""
    query_tokens = tokenize(query)
    query_set = set(query_tokens)
    rows: list[list[float]] = []

    for candidate in candidates:
        chunk = candidate.chunk
        components = candidate.component_scores
        chunk_tokens = set(tokenize(chunk.text))

        overlap = len(query_set & chunk_tokens)
        digits = sum(ch.isdigit() for ch in chunk.text)

        rows.append(
            [
                float(candidate.score),
                float(components.get("dense", 0.0)),
                _rank_inv(components, "dense"),
                float(components.get("bm25", 0.0)),
                _rank_inv(components, "bm25"),
                float(components.get("splade", 0.0)),
                _rank_inv(components, "splade"),
                float(sum(1 for arm in ("dense", "bm25", "splade") if arm in components)),
                float(chunk.token_estimate),
                1.0 if chunk.kind is ChunkKind.TABLE else 0.0,
                float(_SECTION_IDS.get(chunk.section, 0)),
                float(len(query_tokens)),
                float(overlap),
                overlap / len(query_set) if query_set else 0.0,
                digits / max(len(chunk.text), 1),
                float(_CURRENT_YEAR - chunk.fiscal_year),
            ]
        )

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
