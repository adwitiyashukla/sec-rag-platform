"""Typed error hierarchy.

Every failure mode in the platform maps to one of these. The API layer
translates them into HTTP responses in a single place, so handlers never
construct status codes by hand.
"""

from __future__ import annotations


class SecRagError(Exception):
    """Base class for every error raised by this package."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ConfigurationError(SecRagError):
    """Settings are missing or mutually inconsistent."""

    status_code = 500
    code = "configuration_error"


class ProviderError(SecRagError):
    """An upstream LLM provider failed."""

    status_code = 502
    code = "provider_error"


class AllProvidersFailedError(ProviderError):
    """Every provider in the fallback chain failed."""

    code = "all_providers_failed"


class RateLimitError(ProviderError):
    """Provider rejected the request because of rate limiting."""

    status_code = 429
    code = "rate_limited"


class IngestionError(SecRagError):
    """A document could not be fetched or parsed."""

    status_code = 422
    code = "ingestion_error"


class EdgarError(IngestionError):
    """EDGAR rejected or could not satisfy the request."""

    code = "edgar_error"


class RetrievalError(SecRagError):
    """The retrieval stage failed."""

    status_code = 500
    code = "retrieval_error"


class IndexNotReadyError(RetrievalError):
    """A query arrived before any documents were indexed."""

    status_code = 409
    code = "index_not_ready"


class GuardrailTrippedError(SecRagError):
    """A guardrail blocked the request or the response.

    This is not a bug. It is the system correctly refusing to answer, and the
    API surfaces it as a normal response rather than an error.
    """

    status_code = 200
    code = "guardrail_tripped"


class EvaluationError(SecRagError):
    """The evaluation harness could not complete a run."""

    status_code = 500
    code = "evaluation_error"
