"""Domain models shared across every layer.

These are deliberately provider-agnostic and storage-agnostic. Nothing in this
module imports a vector store, an HTTP client, or a model runtime, which keeps
the domain testable in isolation.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Frozen(BaseModel):
    """Immutable, strictly validated base model."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Mutable(BaseModel):
    """Validated but mutable base model, for accumulators such as traces."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Filings and chunks
# ---------------------------------------------------------------------------


class FilingSection(StrEnum):
    """The 10-K items we care about.

    Section identity is a retrieval signal in its own right. A question about
    risk belongs in Item 1A, a question about results belongs in Item 7, and
    scoring a chunk higher when its section matches the query intent measurably
    improves precision.
    """

    BUSINESS = "item_1_business"
    RISK_FACTORS = "item_1a_risk_factors"
    LEGAL_PROCEEDINGS = "item_3_legal_proceedings"
    MDA = "item_7_mda"
    MARKET_RISK = "item_7a_market_risk"
    FINANCIAL_STATEMENTS = "item_8_financial_statements"
    CONTROLS = "item_9a_controls"
    OTHER = "other"


class ChunkKind(StrEnum):
    """Whether a chunk is prose or a serialised table.

    Tables are chunked and prompted differently from narrative text, so the
    distinction has to survive all the way to the generation layer.
    """

    PROSE = "prose"
    TABLE = "table"


class Filing(Frozen):
    """One SEC filing."""

    filing_id: str = Field(description="Stable id, formatted as {cik}-{form}-{fy}")
    cik: str = Field(description="Zero-padded 10 digit Central Index Key")
    ticker: str | None = None
    company: str
    form_type: str = Field(default="10-K")
    fiscal_year: int
    filing_date: date | None = None
    source_url: str

    @model_validator(mode="after")
    def _check_cik(self) -> Self:
        if not self.cik.isdigit():
            msg = f"cik must be numeric, got {self.cik!r}"
            raise ValueError(msg)
        return self


class Chunk(Frozen):
    """A retrievable unit of text with the provenance needed to cite it."""

    chunk_id: str
    filing_id: str
    text: str
    kind: ChunkKind = ChunkKind.PROSE
    section: FilingSection = FilingSection.OTHER

    # Denormalised filing metadata. Duplicated deliberately: it lets the vector
    # store filter without a join, and lets a citation render without a lookup.
    company: str
    ticker: str | None = None
    fiscal_year: int
    source_url: str

    ordinal: int = Field(default=0, description="Position within its section")
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0

    def citation_label(self) -> str:
        """Short human-readable provenance string, for example 'AAPL FY2023 Item 1A'."""
        who = self.ticker or self.company
        section = self.section.value.replace("_", " ").title()
        return f"{who} FY{self.fiscal_year} {section}"

    def contextual_text(self) -> str:
        """Text as it is embedded, prefixed with its provenance.

        A chunk stripped of context is ambiguous: "revenue increased 8%" could
        belong to any company in any year. Prefixing company, year, and section
        before embedding lets the retriever disambiguate across a multi-company
        corpus, and measurably improves precision when several filings discuss
        the same topic. The prefix is embedded but never cited, so quotes shown
        to the user stay verbatim.
        """
        kind = "Table" if self.kind is ChunkKind.TABLE else "Excerpt"
        return f"{self.citation_label()} | {kind}\n{self.text}"


class ScoredChunk(Frozen):
    """A chunk with the score and provenance of the stage that produced it."""

    chunk: Chunk
    score: float
    stage: str = Field(description="dense, sparse, splade, fused, cross_encoder, or ltr")
    rank: int = 0

    # Per-arm scores survive fusion so the learning-to-rank model can use them
    # as features and the trace can explain why a chunk surfaced.
    component_scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------


class QueryIntent(StrEnum):
    """Which pipeline should answer a question.

    Routing exists because these need genuinely different machinery. A numeric
    question should be computed from XBRL facts, not read out of prose, and a
    comparative question needs several retrievals rather than one.
    """

    FACTOID = "factoid"
    NUMERIC = "numeric"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"


class RouteDecision(Frozen):
    """Output of the query router, including why it decided what it did."""

    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    fell_back: bool = Field(
        default=False,
        description="True when confidence was below threshold and we defaulted to factoid",
    )


# ---------------------------------------------------------------------------
# Answers and citations
# ---------------------------------------------------------------------------


class Citation(Frozen):
    """Links one span of the answer to the chunk that supports it."""

    marker: int = Field(ge=1, description="The n in [n] as it appears in the answer")
    chunk_id: str
    label: str
    source_url: str
    quote: str = Field(default="", description="Supporting span from the source chunk")
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)


class AnswerStatus(StrEnum):
    OK = "ok"
    REFUSED_UNGROUNDED = "refused_ungrounded"
    REFUSED_NO_CONTEXT = "refused_no_context"
    REFUSED_GUARDRAIL = "refused_guardrail"


class Answer(Frozen):
    """A generated answer plus everything needed to audit it."""

    text: str
    status: AnswerStatus = AnswerStatus.OK
    citations: list[Citation] = Field(default_factory=list)
    groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_reason: str | None = None


class NumericResult(Frozen):
    """A figure computed from XBRL facts rather than generated by a model.

    Carrying the formula and inputs is what makes the number auditable. A user
    can check the arithmetic without trusting the language model at all.
    """

    label: str
    value: float | None
    unit: str = "USD"
    formula: str = ""
    inputs: dict[str, float] = Field(default_factory=dict)
    period: str = ""
    concept: str = ""
    source_url: str = ""


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class QueryRequest(Mutable):
    question: Annotated[str, Field(min_length=3, max_length=2000)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 6
    companies: list[str] = Field(
        default_factory=list, description="Optional ticker filter, for example ['AAPL']"
    )
    fiscal_years: list[int] = Field(default_factory=list)
    sections: list[FilingSection] = Field(default_factory=list)
    use_cache: bool = True
    use_reranker: bool = True
    reranker: str = Field(default="cross_encoder", description="cross_encoder, ltr, or none")


class QueryResponse(Mutable):
    question: str
    answer: Answer
    route: RouteDecision | None = None
    contexts: list[ScoredChunk] = Field(default_factory=list)
    numeric_results: list[NumericResult] = Field(default_factory=list)
    trace_id: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
