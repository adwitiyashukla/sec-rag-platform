# ADR 0005: Ship a deterministic offline LLM provider

Status: accepted

## Context

Tests that call a real language model are slow, cost money or quota, fail when
someone else's rate limit is hit, and produce different output every run. Tests
that mock the model with a fixed string exercise none of the code that matters:
citation parsing, groundedness verification, refusal logic.

CI also has no API keys, by design.

## Decision

Implement `EchoProvider`, which parses the retrieved context out of the prompt
and composes a genuinely grounded, correctly cited extractive answer from it. It
also returns schema-valid JSON for the structured call sites.

## Consequences

The full pipeline runs in CI with no key, no network, and no flakiness, and a
green run means something: citation markers are real, groundedness is computed
over real passages, and refusal paths are exercised.

`build_chain` falls back to this provider when no credentials are configured, so
a misconfigured deployment starts and is inspectable rather than refusing to
boot. That is a deliberate choice to degrade rather than fail, appropriate for a
demo service and worth revisiting for anything handling real traffic.

The provider is not a quality benchmark. Its answers are extractive by
construction, so generation metrics measured against it describe the plumbing,
not the model. Absolute answer quality is measured separately with a real
provider configured.
