"""Retrieval and generation metrics.

Implemented directly rather than pulled from a framework, for two reasons.
Ranking metrics have subtle conventions that matter (what counts as relevant,
how ties break, what happens when a query has no relevant document at all), and
a metric whose definition you cannot see is a metric you cannot defend in a
review. Second, every metric here is deterministic and needs no model call, so
the whole suite runs in CI in seconds with no API key and no flaky network.

Relevance judgement is weak supervision, and it is worth being upfront about
that: rather than hand-labelling every chunk, a chunk counts as relevant when it
comes from the expected 10-K Item and contains at least one expected term. That
is a proxy, not ground truth. It is stable, reproducible, and good enough to
detect regressions, which is what the CI gate needs it for.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from secrag.core.types import ScoredChunk


def is_relevant(
    scored: ScoredChunk, expected_sections: Sequence[str], expected_terms: Sequence[str]
) -> bool:
    """Weak relevance judgement for one retrieved chunk."""
    if expected_sections and scored.chunk.section.value not in expected_sections:
        return False
    if not expected_terms:
        return True
    lowered = scored.chunk.text.lower()
    return any(term.lower() in lowered for term in expected_terms)


def relevance_vector(
    results: Sequence[ScoredChunk],
    expected_sections: Sequence[str],
    expected_terms: Sequence[str],
) -> list[int]:
    return [1 if is_relevant(r, expected_sections, expected_terms) else 0 for r in results]


# --------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------


def hit_rate_at_k(relevance: Sequence[int], k: int) -> float:
    """1.0 if any relevant document appears in the top k."""
    return 1.0 if any(relevance[:k]) else 0.0


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    window = relevance[:k]
    return sum(window) / len(window) if window else 0.0


def recall_at_k(relevance: Sequence[int], k: int, total_relevant: int | None = None) -> float:
    """Recall against relevant documents found anywhere in the result list.

    Exhaustive labels do not exist for this corpus, so the denominator is the
    number of relevant documents the system retrieved at all. This measures
    ranking quality rather than absolute recall, and it is only compared
    against itself across runs, which is what a regression gate needs.
    """
    denominator = total_relevant if total_relevant is not None else sum(relevance)
    return sum(relevance[:k]) / denominator if denominator else 0.0


def reciprocal_rank(relevance: Sequence[int]) -> float:
    for index, value in enumerate(relevance, start=1):
        if value:
            return 1.0 / index
    return 0.0


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevance[:k], start=1))


def ndcg_at_k(relevance: Sequence[int], k: int) -> float:
    """Normalised DCG, rewarding relevant documents ranked higher."""
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    return dcg_at_k(relevance, k) / ideal if ideal > 0 else 0.0


# --------------------------------------------------------------------------
# Generation metrics
# --------------------------------------------------------------------------

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def citation_validity(answer_text: str, n_contexts: int) -> float:
    """Fraction of citation markers that point at a real context.

    A model that invents [7] when six passages were supplied has produced an
    unverifiable citation, which is worse than no citation because it looks
    checkable.
    """
    markers = [int(m) for m in _MARKER_RE.findall(answer_text)]
    if not markers:
        return 0.0
    valid = sum(1 for m in markers if 1 <= m <= n_contexts)
    return valid / len(markers)


def citation_density(answer_text: str) -> float:
    """Fraction of claim spans that carry a citation.

    Measured per claim span rather than per sentence, matching how
    groundedness is scored. A per-sentence count penalises the ordinary and
    correct habit of stating a few sentences and citing once at the end, so it
    reports low numbers for answers that are fully attributed.
    """
    from secrag.generation.grounding import split_claims

    claims = split_claims(answer_text)
    if not claims:
        return 0.0
    cited = sum(1 for _, markers in claims if markers)
    return cited / len(claims)


def answer_contains(answer_text: str, required: Sequence[str]) -> float:
    """Fraction of required terms that appear in the answer."""
    if not required:
        return 1.0
    lowered = answer_text.lower()
    return sum(1 for term in required if term.lower() in lowered) / len(required)


def numeric_accuracy(
    actual: float | None, expected: float | None, tolerance_pct: float = 1.0
) -> float | None:
    """1.0 when a computed figure matches the expected value within tolerance.

    Returns None when the case is not numeric, so it can be excluded from the
    average rather than counted as a zero.
    """
    if expected is None:
        return None
    if actual is None:
        return 0.0
    if expected == 0:
        return 1.0 if abs(actual) < 1e-9 else 0.0
    return 1.0 if abs(actual - expected) / abs(expected) * 100.0 <= tolerance_pct else 0.0


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MetricAccumulator:
    """Collects per-case values and reports means, skipping None."""

    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float | None) -> None:
        if value is None:
            return
        self.values.setdefault(name, []).append(float(value))

    def mean(self, name: str) -> float:
        series = self.values.get(name, [])
        return sum(series) / len(series) if series else 0.0

    def count(self, name: str) -> int:
        return len(self.values.get(name, []))

    def summary(self) -> dict[str, float]:
        return {name: round(self.mean(name), 4) for name in sorted(self.values)}
