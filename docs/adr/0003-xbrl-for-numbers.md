# ADR 0003: Answer numeric questions from XBRL, not from retrieved text

Status: accepted

## Context

Language models are unreliable at reading figures out of 10-K tables. The tables
are large, span multiple periods, nest headers, and carry scaling captions ("in
millions, except per share data") that sit far from the figures they govern.
Asked for revenue growth, a model will typically produce a plausible and wrong
number, which on financial data is the worst possible failure: confident,
specific, and unverifiable at a glance.

The SEC publishes the same figures as structured XBRL through the
`companyfacts` API.

## Decision

Route numeric questions to a deterministic engine that looks the fact up and
computes the arithmetic in pandas. Retrieval still runs and still supplies
narrative context, but the number itself never comes from the model. Every
computed figure carries its formula and its inputs so it can be audited.

## Consequences

Figures become verifiable rather than plausible. The answer to "what was Apple's
FY2024 gross margin" is 46.21 percent, accompanied by gross profit 180,683
million, revenue 391,035 million, and the formula that combines them.

This introduced its own bug, worth recording because it is subtle and silent.
Each XBRL observation carries `fp` and `fy` fields, but those describe the
*report* the fact appeared in, not the period the fact covers. A 10-K also tags
its fourth-quarter figures, and those carry `fp="FY"` and `form="10-K"` too.
Filtering on those fields alone admits quarterly values as annual ones, and
Apple's FY2020 revenue reads as 64.7 billion (its Q4 figure) instead of 274.5
billion. Every growth rate computed from it is then wrong while looking
entirely reasonable.

The fix is to select annual facts by measuring the actual start-to-end duration
of each observation and accepting 340 to 400 days. Fiscal years are labelled by
the period midpoint so that a fiscal year closing in January is not shifted
forward. Both behaviours are covered by tests.

The cost is a hand-maintained mapping from natural language metrics to US-GAAP
concepts, with fallback chains for companies that tag the same concept
differently. This is rule-based on purpose: "gross margin means gross profit
over revenue" is a definition, not a prediction, and learning it would add a
failure mode to something with an exact answer.
