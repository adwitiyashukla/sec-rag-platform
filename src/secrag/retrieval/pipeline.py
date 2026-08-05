"""Hybrid retrieval pipeline.

Three arms run against the same corpus and are fused by rank, then a reranker
reorders the survivors:

    query -> [ dense | bm25 | splade ] -> RRF -> rerank -> top_k

Each arm fails differently, which is the reason to keep all three. Dense
retrieval understands paraphrase but blurs near-identical identifiers. BM25 is
exact but blind to synonyms. SPLADE sits between the two, expanding terms in
learned rather than lexical space. Fusing them recovers documents that any one
arm alone would have missed.

Arms can be enabled individually, which is what makes the ablation table in the
README a measurement rather than a claim.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import ScoredChunk
from secrag.observability.tracing import span
from secrag.retrieval.bm25 import BM25Index
from secrag.retrieval.embedder import Embedder
from secrag.retrieval.fusion import reciprocal_rank_fusion
from secrag.retrieval.rerank import (
    CrossEncoderReranker,
    LTRReranker,
    Reranker,
    build_reranker,
)
from secrag.retrieval.store import SearchFilter, VectorStore

log = get_logger(__name__)

ALL_ARMS = ("dense", "bm25", "splade")


@dataclass(slots=True)
class RetrievalResult:
    """Final ranking plus the intermediate state needed to explain it."""

    chunks: list[ScoredChunk]
    arms: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    fused: list[ScoredChunk] = field(default_factory=list)
    reranker: str = "none"

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class HybridRetriever:
    """Owns the retrieval arms, fusion, and reranking."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self.store = store or VectorStore(self.settings, self.embedder)
        self.bm25 = bm25 or BM25Index(self.settings)
        self._rerankers: dict[str, Reranker] = {}
        # Same reasoning as the store and the embedder: warmup runs on a
        # worker thread alongside live requests, so building the BM25 index
        # and constructing rerankers are both genuinely concurrent.
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    def ensure_ready(self, *, force: bool = False) -> int:
        """Build the BM25 index from the vector store.

        Derived from the store rather than persisted separately, so the lexical
        and vector views of the corpus cannot drift apart.
        """
        if self.bm25.is_ready and not force:
            return self.bm25.size
        with self._lock:
            if self.bm25.is_ready and not force:
                return self.bm25.size
            return self.bm25.build(self.store.iter_chunks())

    def reranker(self, name: str) -> Reranker:
        if name not in self._rerankers:
            with self._lock:
                if name not in self._rerankers:
                    self._rerankers[name] = build_reranker(name, self.settings)
        return self._rerankers[name]

    @property
    def corpus_size(self) -> int:
        return self.store.count()

    # -- retrieval --------------------------------------------------------

    def run_arms(
        self,
        query: str,
        flt: SearchFilter | None = None,
        arms: Sequence[str] = ALL_ARMS,
    ) -> dict[str, list[ScoredChunk]]:
        """Run each enabled retrieval arm independently."""
        results: dict[str, list[ScoredChunk]] = {}

        if "dense" in arms:
            vector = self.embedder.embed_query(query)
            results["dense"] = self.store.search_dense(vector, self.settings.dense_top_k, flt)

        if "bm25" in arms:
            self.ensure_ready()
            results["bm25"] = self.bm25.search(query, self.settings.sparse_top_k, flt)

        if "splade" in arms and self.settings.enable_splade:
            sparse = self.embedder.embed_sparse_query(query)
            results["splade"] = self.store.search_sparse(sparse, self.settings.splade_top_k, flt)

        return {arm: hits for arm, hits in results.items() if hits}

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        flt: SearchFilter | None = None,
        arms: Sequence[str] = ALL_ARMS,
        reranker: str = "cross_encoder",
        weights: dict[str, float] | None = None,
    ) -> RetrievalResult:
        """Full pipeline: retrieve, fuse, rerank."""
        top_k = top_k or self.settings.rerank_top_n

        with span("retrieve", query_chars=len(query), arms=list(arms), reranker=reranker):
            arm_results = self.run_arms(query, flt, arms)
            if not arm_results:
                log.info("retrieval_empty", query=query[:80])
                return RetrievalResult(chunks=[], arms={}, fused=[], reranker=reranker)

            # With a single arm there is nothing to fuse, and running RRF anyway
            # would replace real scores with reciprocal ranks for no benefit.
            if len(arm_results) == 1:
                fused = list(next(iter(arm_results.values())))[: self.settings.rerank_candidates]
            else:
                fused = reciprocal_rank_fusion(
                    arm_results,
                    k=self.settings.rrf_k,
                    weights=weights,
                    limit=self.settings.rerank_candidates,
                )

            engine = self.reranker(reranker)
            ranked = engine.rerank(query, fused, top_k)

        log.info(
            "retrieved",
            arms={arm: len(hits) for arm, hits in arm_results.items()},
            fused=len(fused),
            returned=len(ranked),
            reranker=engine.name,
        )
        return RetrievalResult(chunks=ranked, arms=arm_results, fused=fused, reranker=engine.name)

    def warmup(self) -> None:
        """Load every model and index before the first request arrives.

        Every reranker is loaded, not just the default one. Warming only the
        cross-encoder left the first learning-to-rank request paying the model
        load itself, which measured around five seconds against roughly 150 ms
        once warm. A user switching reranker in the UI met that delay and had
        no way to know it was one-time.
        """
        self.embedder.warmup()
        self.ensure_ready()

        for name in ("cross_encoder", "ltr"):
            engine = self.reranker(name)
            if isinstance(engine, CrossEncoderReranker):
                engine.warmup()
            elif isinstance(engine, LTRReranker):
                engine.booster  # noqa: B018 - touching the property forces the load

        log.info(
            "retriever_warm",
            corpus=self.corpus_size,
            bm25=self.bm25.size,
            rerankers=sorted(self._rerankers),
        )
