# ADR 0002: Fuse retrieval arms by rank, not by score

Status: accepted

## Context

Dense retrieval fails on precise identifiers. An embedding model places "Item
9A" and "Item 9B" at nearly the same point, and financial filings are full of
defined terms, statute references, and tickers where exactness matters. Lexical
BM25 handles those well but is blind to paraphrase.

Combining them requires a fusion rule. The tempting approach is to normalise
scores onto a common scale and take a weighted sum.

## Decision

Use Reciprocal Rank Fusion:

    RRF(d) = sum over arms of weight / (k + rank(d))

with k = 60 following Cormack et al. (2009), exposed as a setting.

## Consequences

Score normalisation would require assumptions that do not hold. BM25 scores are
unbounded and corpus-dependent, cosine similarity is bounded to [-1, 1] and
tightly clustered near the top, and SPLADE scores are on a third scale entirely.
Min-max normalising them makes the result depend on the score range of whichever
documents happened to be retrieved, which changes per query.

Ranks are directly comparable and need no assumptions. The tradeoff is that RRF
discards magnitude: a document that BM25 scored 40.0 and one it scored 4.0 count
the same if they are ranked adjacently. In practice, agreement across
independent arms is a stronger relevance signal than any single arm's magnitude.

There is a test asserting this property directly: multiplying every BM25 score
by 100,000 must not change the fused ordering.

Component scores are carried through fusion rather than discarded, because the
learning-to-rank reranker consumes them as features and the trace uses them to
explain why a passage surfaced.
