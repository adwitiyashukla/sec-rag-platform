"""Answer synthesis.

Sequence, and the reason for each step:

    contexts -> injection scan -> prompt -> LLM -> PII redaction
             -> groundedness verification -> accept or refuse

The verification step runs after generation rather than before, because the
thing being checked is the model's output, not its input. A model that produces
a fluent answer citing passages that do not support it has failed in the way
that matters most on financial data, and no amount of prompt engineering
detects that reliably. Measuring it does.

Refusing is treated as a successful outcome, not an error. On filings, "the
filing does not state this" is frequently the correct answer, and a system that
cannot say it will invent one instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import Answer, AnswerStatus, NumericResult, ScoredChunk
from secrag.generation.grounding import GroundingReport, verify
from secrag.generation.prompts import REFUSAL_NO_CONTEXT, SYSTEM_PROMPT, build_user_prompt
from secrag.guardrails import injection, pii
from secrag.observability.tracing import span
from secrag.providers.base import ChatMessage, LLMProvider
from secrag.providers.registry import build_chain
from secrag.retrieval.embedder import Embedder

log = get_logger(__name__)


@dataclass(slots=True)
class GenerationResult:
    """The answer plus everything needed to audit how it was produced."""

    answer: Answer
    contexts: list[ScoredChunk] = field(default_factory=list)
    injection: injection.InjectionReport | None = None
    grounding: GroundingReport | None = None
    redacted: dict[str, int] = field(default_factory=dict)


class Generator:
    """Turns retrieved context into a cited, verified answer."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or build_chain(self.settings)
        self.embedder = embedder or Embedder(self.settings)

    # -- helpers ----------------------------------------------------------

    def _prepare(
        self, contexts: Sequence[ScoredChunk]
    ) -> tuple[list[ScoredChunk], injection.InjectionReport | None]:
        """Scan retrieved content and drop anything carrying instructions."""
        if not self.settings.enable_injection_detection:
            return list(contexts), None
        report = injection.scan_contexts(contexts)
        return injection.drop_flagged(contexts, report), report

    def _messages(
        self,
        question: str,
        contexts: Sequence[ScoredChunk],
        numeric: Sequence[NumericResult],
    ) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_user_prompt(question, contexts, numeric)),
        ]

    def _finalise(
        self,
        raw_text: str,
        contexts: Sequence[ScoredChunk],
        injection_report: injection.InjectionReport | None,
    ) -> GenerationResult:
        """Redact, verify, and decide whether the answer is fit to return."""
        redaction = (
            pii.redact(raw_text)
            if self.settings.enable_pii_redaction
            else pii.RedactionReport(text=raw_text)
        )
        grounding = verify(redaction.text, contexts, self.embedder)

        status = AnswerStatus.OK
        refusal: str | None = None
        text = redaction.text

        if not grounding.has_citations:
            status = AnswerStatus.REFUSED_UNGROUNDED
            refusal = "The generated answer cited no sources, so it could not be verified."
        elif grounding.groundedness < self.settings.min_groundedness:
            status = AnswerStatus.REFUSED_UNGROUNDED
            refusal = (
                f"Support for this answer scored {grounding.groundedness:.2f}, below the "
                f"required {self.settings.min_groundedness:.2f}. The retrieved passages do "
                "not clearly support the claims made, so the answer is withheld."
            )

        if status is not AnswerStatus.OK:
            log.warning(
                "answer_refused",
                status=status.value,
                groundedness=grounding.groundedness,
                citations=len(grounding.citations),
            )

        return GenerationResult(
            answer=Answer(
                text=text,
                status=status,
                citations=grounding.citations,
                groundedness=grounding.groundedness,
                refusal_reason=refusal,
            ),
            contexts=list(contexts),
            injection=injection_report,
            grounding=grounding,
            redacted=redaction.counts,
        )

    @staticmethod
    def _no_context() -> GenerationResult:
        return GenerationResult(
            answer=Answer(
                text=REFUSAL_NO_CONTEXT,
                status=AnswerStatus.REFUSED_NO_CONTEXT,
                refusal_reason="Retrieval returned no passages for this question.",
            )
        )

    # -- generation -------------------------------------------------------

    async def generate(
        self,
        question: str,
        contexts: Sequence[ScoredChunk],
        numeric: Sequence[NumericResult] = (),
    ) -> GenerationResult:
        if not contexts:
            return self._no_context()

        used, injection_report = self._prepare(contexts)
        with span("generate", contexts=len(used)):
            completion = await self.provider.complete(
                self._messages(question, used, numeric),
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
        return self._finalise(completion.text, used, injection_report)

    async def stream(
        self,
        question: str,
        contexts: Sequence[ScoredChunk],
        numeric: Sequence[NumericResult] = (),
    ) -> AsyncIterator[str]:
        """Yield answer tokens as they arrive.

        Verification cannot run until the answer is complete, so a streamed
        response is necessarily unverified while it is being produced. The
        caller is expected to call finalise_streamed once the stream ends and
        to present the verdict alongside the text.
        """
        if not contexts:
            yield REFUSAL_NO_CONTEXT
            return

        used, _ = self._prepare(contexts)
        with span("generate_stream", contexts=len(used)):
            async for piece in self.provider.stream(
                self._messages(question, used, numeric),
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            ):
                yield piece

    def finalise_streamed(self, text: str, contexts: Sequence[ScoredChunk]) -> GenerationResult:
        """Verify a fully streamed answer."""
        if not contexts:
            return self._no_context()
        used, injection_report = self._prepare(contexts)
        return self._finalise(text, used, injection_report)
