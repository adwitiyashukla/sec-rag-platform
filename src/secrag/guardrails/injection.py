"""Indirect prompt injection detection.

The threat model is specific. A user's question is not the dangerous input,
because the user only ever harms themselves. The dangerous input is retrieved
content, which is attacker-controlled in any system that indexes documents it
did not author, and which the model reads with the same trust as instructions.

SEC filings are a low risk corpus, but the ingestion path accepts arbitrary
HTML, and the pattern generalises to every RAG system that ever accepts an
upload. Detection runs over retrieved chunks before they reach the prompt.

This is heuristic and defence in depth, not a guarantee. The structural
mitigations matter more: retrieved text is delimited and labelled as data, the
system prompt states that context is never instructions, and every claim is
verified against its source afterwards.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from secrag.core.logging import get_logger
from secrag.core.types import ScoredChunk

log = get_logger(__name__)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
            r"\b(instruction|prompt|rule|direction|context)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\byou\s+are\s+(now|no\s+longer)\b|\bact\s+as\s+(if|a|an)\b"
            r"|\bpretend\s+(to\s+be|you)\b|\bnew\s+(persona|role|identity)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fake_turn_boundary",
        re.compile(
            r"^\s*(system|assistant|user)\s*[:>]|<\|?(im_start|im_end|system|endoftext)\|?>",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(reveal|print|repeat|output|show)\b[^.\n]{0,30}"
            r"\b(system\s+prompt|instructions|api[_\s]?key|secret|credential)",
            re.IGNORECASE,
        ),
    ),
    (
        "citation_tampering",
        re.compile(
            r"\b(do\s+not|don.t|never)\b[^.\n]{0,20}\b(cite|citation|source|reference)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(slots=True)
class InjectionFinding:
    chunk_id: str
    label: str
    pattern: str
    excerpt: str


@dataclass(slots=True)
class InjectionReport:
    findings: list[InjectionFinding] = field(default_factory=list)
    scanned: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.findings

    @property
    def flagged_chunk_ids(self) -> set[str]:
        return {f.chunk_id for f in self.findings}


def scan_text(text: str) -> list[tuple[str, str]]:
    """Return (pattern_name, excerpt) for each pattern that fires."""
    hits: list[tuple[str, str]] = []
    for name, pattern in _PATTERNS:
        if match := pattern.search(text):
            start = max(0, match.start() - 30)
            hits.append((name, text[start : match.end() + 30].replace("\n", " ")))
    return hits


def scan_contexts(contexts: Sequence[ScoredChunk]) -> InjectionReport:
    """Scan retrieved chunks for injected instructions."""
    report = InjectionReport(scanned=len(contexts))
    for scored in contexts:
        for name, excerpt in scan_text(scored.chunk.text):
            report.findings.append(
                InjectionFinding(
                    chunk_id=scored.chunk.chunk_id,
                    label=scored.chunk.citation_label(),
                    pattern=name,
                    excerpt=excerpt,
                )
            )
    if report.findings:
        log.warning(
            "injection_detected",
            count=len(report.findings),
            patterns=sorted({f.pattern for f in report.findings}),
        )
    return report


def drop_flagged(contexts: Sequence[ScoredChunk], report: InjectionReport) -> list[ScoredChunk]:
    """Remove flagged chunks, unless that would empty the context entirely.

    Dropping every passage turns a suspicious answer into no answer at all. If
    everything is flagged the passages are kept and the caller is expected to
    surface the warning instead, which is more useful than a blank refusal.
    """
    if report.is_clean:
        return list(contexts)
    flagged = report.flagged_chunk_ids
    kept = [c for c in contexts if c.chunk.chunk_id not in flagged]
    return kept or list(contexts)
