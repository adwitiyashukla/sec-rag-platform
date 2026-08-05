"""Embedding.

Runs BGE and SPLADE through ONNX Runtime rather than PyTorch. The weights are
identical; the difference is that the ONNX path removes a 2.5 GB torch
dependency, drops container cold start from roughly 45 seconds to about 5, and
runs comfortably on the free CPU tiers this project targets. That tradeoff is
what makes the deployment viable at zero cost.

Models load lazily and are cached per process, so importing this module is free
and a test that never embeds anything never downloads a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.observability.tracing import span

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A learned sparse vector, decoupled from the fastembed types."""

    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices


class Embedder:
    """Dense and learned-sparse embeddings over a shared model cache."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dense: Any | None = None
        self._sparse: Any | None = None

    # -- lazy model handles ----------------------------------------------

    @property
    def dense(self) -> Any:
        if self._dense is None:
            from fastembed import TextEmbedding

            with span("load_dense_model", model=self.settings.dense_model):
                self._dense = TextEmbedding(
                    model_name=self.settings.dense_model,
                    cache_dir=str(self.settings.models_dir),
                )
            log.info("dense_model_loaded", model=self.settings.dense_model)
        return self._dense

    @property
    def sparse(self) -> Any:
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            with span("load_sparse_model", model=self.settings.sparse_model):
                self._sparse = SparseTextEmbedding(
                    model_name=self.settings.sparse_model,
                    cache_dir=str(self.settings.models_dir),
                )
            log.info("sparse_model_loaded", model=self.settings.sparse_model)
        return self._sparse

    # -- dense ------------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed passages. Returns an (n, dim) L2-normalised float32 array."""
        if not texts:
            return np.zeros((0, self.settings.dense_dim), dtype=np.float32)
        with span("embed_documents", count=len(texts)):
            vectors = list(self.dense.embed(list(texts), batch_size=self.settings.embed_batch_size))
        return _normalise(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a query, applying BGE's asymmetric query prefix.

        BGE v1.5 was trained with an instruction prefix on the query side only.
        Skipping it is a silent correctness bug: retrieval still returns results,
        they are simply measurably worse. Applying it explicitly here rather than
        relying on the library keeps the behaviour visible and version-stable.
        """
        prefixed = f"{self.settings.dense_query_prefix}{text}"
        with span("embed_query"):
            vector = next(iter(self.dense.embed([prefixed])))
        normalised: NDArray[np.float32] = _normalise(
            np.asarray(vector, dtype=np.float32).reshape(1, -1)
        )[0]
        return normalised

    # -- sparse -----------------------------------------------------------

    def embed_sparse_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []
        with span("embed_sparse_documents", count=len(texts)):
            raw = list(self.sparse.embed(list(texts), batch_size=self.settings.embed_batch_size))
        return [_to_sparse(item) for item in raw]

    def embed_sparse_query(self, text: str) -> SparseVector:
        with span("embed_sparse_query"):
            raw = next(iter(self.sparse.embed([text])))
        return _to_sparse(raw)

    # -- lifecycle --------------------------------------------------------

    def warmup(self, *, include_sparse: bool = True) -> None:
        """Force model load and one inference pass.

        Called at service startup so the first user request does not absorb the
        model download and ONNX graph initialisation.
        """
        self.embed_query("warmup")
        if include_sparse and self.settings.enable_splade:
            self.embed_sparse_query("warmup")
        log.info("embedder_warm", dense=self.settings.dense_model)


def _normalise(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalise rows so a dot product is exactly cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)


def _to_sparse(item: Any) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in item.indices],
        values=[float(v) for v in item.values],
    )
