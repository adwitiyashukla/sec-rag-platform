# ADR 0004: Verify groundedness without a second model call

Status: accepted

## Context

An answer that cites [2] is not thereby supported by [2]. Something has to check
the link, or citations are decoration.

The common approach is LLM-as-judge: send the answer and its sources to a model
and ask whether each claim is supported.

## Decision

Compute groundedness deterministically. Split the answer into sentences, extract
citation markers, and compare each sentence against the passages it cites using
embedding cosine similarity. The supporting quote shown to the user is selected
by lexical overlap, which is cheap and only decides which span to display.

## Consequences

Three reasons this beats a judge model here:

1. **Latency and quota.** A judge doubles both. On a free tier of 30 requests
   per minute, that halves effective capacity.
2. **Independence.** A model grading its own output, especially the same model,
   is a weak check.
3. **Testability.** A deterministic score can be asserted on in CI. The
   evaluation suite runs with no API key at all, and its groundedness numbers
   are reproducible rather than resampled each run.

The limitation is that embedding similarity measures topical relatedness, not
entailment. A sentence that contradicts its source can still score highly if it
discusses the same subject with the same vocabulary. This catches unsupported
and off-topic claims, not subtle factual inversion.

That gap is covered from the other direction: numeric claims, where inversion
would matter most, do not come from the model at all (see ADR 0003).

The threshold of 0.35 for a supported sentence was calibrated against observed
similarities on this corpus, where genuinely supported sentences sit around 0.6
to 0.8. Answers scoring below `SECRAG_MIN_GROUNDEDNESS` are withheld with an
explanation rather than returned with a warning.
