"""Rank fusion.

Reciprocal Rank Fusion combines rankings by position rather than by score,
which is the whole point: BM25 scores are unbounded, cosine similarity is
bounded to [-1, 1], and SPLADE scores are something else again. Normalising
those onto a common scale requires assumptions that do not hold. Ranks are
directly comparable with no assumptions at all.

    RRF(d) = sum over arms of 1 / (k + rank(d))

k dampens the influence of the very top positions. The value 60 comes from
Cormack et al. (2009) and is used here as a documented default rather than a
magic number, and it is exposed as a tunable in settings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from secrag.core.types import ScoredChunk
from secrag.observability.tracing import span


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Fuse several ranked lists into one.

    Component scores are carried through to the fused result. They are what the
    learning-to-rank reranker uses as features, and what the trace shows to
    explain why a chunk surfaced.
    """
    with span("rrf_fusion", arms=len(ranked_lists)):
        fused: dict[str, float] = {}
        best: dict[str, ScoredChunk] = {}
        components: dict[str, dict[str, float]] = {}

        for arm, results in ranked_lists.items():
            weight = (weights or {}).get(arm, 1.0)
            # Rank is the position in the list this arm returned, not the rank
            # field carried on the item. The list is the authoritative ordering;
            # a stale rank attribute on a filtered or re-sliced result would
            # otherwise corrupt the fusion silently.
            for rank, scored in enumerate(results, start=1):
                chunk_id = scored.chunk.chunk_id
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
                components.setdefault(chunk_id, {})[arm] = scored.score
                components[chunk_id][f"{arm}_rank"] = float(rank)
                best.setdefault(chunk_id, scored)

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        if limit is not None:
            ordered = ordered[:limit]

        return [
            ScoredChunk(
                chunk=best[chunk_id].chunk,
                score=score,
                stage="fused",
                rank=rank,
                component_scores=components.get(chunk_id, {}),
            )
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        ]
