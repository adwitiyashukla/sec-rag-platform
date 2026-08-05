"""Chunking.

Chunk boundaries decide what the retriever can ever find, so this is one of the
highest-leverage components in a RAG system and one of the most commonly done
badly. Three decisions matter here:

1. Prose splits on sentence boundaries, never mid-sentence. A chunk that begins
   halfway through a clause embeds poorly and reads worse when cited.
2. Tables that must be split repeat their header row in every part. A fragment
   of a financial table without column headers is not just less useful, it is
   actively misleading, because the model cannot tell which period a figure
   belongs to.
3. Chunk ids are deterministic hashes of their content, so re-ingesting a
   filing is idempotent rather than duplicating the corpus.
"""

from __future__ import annotations

import hashlib
import re

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import Chunk, ChunkKind, Filing
from secrag.ingest.parser import Block

log = get_logger(__name__)

# Sentence splitter tuned for filings. The negative lookbehind keeps common
# financial abbreviations from being treated as sentence ends, which otherwise
# shreds passages such as "Inc. reported" or "U.S. GAAP requires".
# Each lookbehind must include the trailing period. The split point sits
# *after* that period, so a guard written as (?<!\bU\.S) inspects ".S." and
# never fires. The bug is silent: "U.S. GAAP requires disclosure" quietly
# becomes two chunks, one of which is the fragment "U.S.".
_ABBREV = (
    r"(?<!\bInc\.)(?<!\bCorp\.)(?<!\bLtd\.)(?<!\bCo\.)(?<!\bNo\.)"
    r"(?<!\bU\.S\.)(?<!\bApprox\.)(?<!\bFig\.)(?<!\bLLC\.)(?<!\bPLC\.)"
    r"(?<!\bJr\.)(?<!\bSr\.)(?<!\bvs\.)(?<!\bNos\.)"
)
_SENTENCE_RE = re.compile(rf"{_ABBREV}(?<=[.!?])[\"')\]]*\s+(?=[A-Z(\"'\[])")

_MIN_CHUNK_CHARS = 120
# A chunk this short carries no retrievable meaning. These come from stray
# headings, page numbers, and signature fragments, and they cost a retrieval
# slot without ever being able to answer anything.
_MIN_CHUNK_TOKENS = 20


def estimate_tokens(text: str) -> int:
    """Character-based token estimate.

    Deliberately not a real tokenizer: chunking runs over every block of every
    filing, and loading a tokenizer here would dominate ingestion time for an
    accuracy gain that does not change where the boundaries land.
    """
    return max(1, len(text) // 4)


def _chunk_id(filing_id: str, section: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(f"{filing_id}|{section}|{ordinal}|{text[:256]}".encode()).hexdigest()
    return digest[:20]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _pack_sentences(sentences: list[str], target: int, overlap: int) -> list[str]:
    """Greedily pack sentences into windows with a sentence-aligned overlap."""
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        tokens = estimate_tokens(sentence)

        # A single sentence longer than the window is split on whitespace as a
        # last resort. Rare in filings, but it must not silently drop text.
        if tokens > target * 1.5:
            if current:
                windows.append(" ".join(current))
                current, current_tokens = [], 0
            words = sentence.split()
            step = max(1, target * 4 // 6)
            for i in range(0, len(words), step):
                windows.append(" ".join(words[i : i + step]))
            continue

        if current_tokens + tokens > target and current:
            windows.append(" ".join(current))
            # Carry back trailing sentences to form the overlap.
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev)
                if carried + prev_tokens > overlap:
                    break
                carry.insert(0, prev)
                carried += prev_tokens
            current, current_tokens = carry, carried

        current.append(sentence)
        current_tokens += tokens

    if current:
        windows.append(" ".join(current))
    return [w for w in windows if len(w) >= _MIN_CHUNK_CHARS or len(windows) == 1]


def _split_table(rendered: str, max_tokens: int) -> list[str]:
    """Split a large table by rows, repeating the header in each part."""
    rows = [r for r in rendered.split("\n") if r.strip()]
    if estimate_tokens(rendered) <= max_tokens or len(rows) <= 2:
        return [rendered]

    header, body = rows[0], rows[1:]
    header_tokens = estimate_tokens(header)
    parts: list[str] = []
    current: list[str] = []
    current_tokens = header_tokens

    for row in body:
        row_tokens = estimate_tokens(row)
        if current_tokens + row_tokens > max_tokens and current:
            parts.append("\n".join([header, *current]))
            current, current_tokens = [], header_tokens
        current.append(row)
        current_tokens += row_tokens

    if current:
        parts.append("\n".join([header, *current]))
    return parts


def _group_blocks(blocks: list[Block]) -> list[Block]:
    """Merge consecutive prose blocks that share a section.

    The parser emits paragraph-level blocks so section boundaries land exactly.
    Packing each paragraph separately would produce many undersized chunks, so
    neighbours in the same section are joined back into one passage first.
    Tables are never merged, since each is a self-contained unit.
    """
    merged: list[Block] = []
    for block in blocks:
        if (
            block.kind is ChunkKind.PROSE
            and merged
            and merged[-1].kind is ChunkKind.PROSE
            and merged[-1].section is block.section
        ):
            previous = merged[-1]
            previous.text = f"{previous.text} {block.text}".strip()
            previous.end = block.end
            continue
        merged.append(
            Block(
                kind=block.kind,
                text=block.text,
                order=len(merged),
                start=block.start,
                end=block.end,
                section=block.section,
            )
        )
    return merged


def chunk_blocks(
    blocks: list[Block], filing: Filing, settings: Settings | None = None
) -> list[Chunk]:
    """Turn parsed blocks into retrievable chunks."""
    settings = settings or get_settings()
    chunks: list[Chunk] = []
    per_section: dict[str, int] = {}

    for block in _group_blocks(blocks):
        section_key = block.section.value
        pieces = (
            _split_table(block.text, settings.max_table_tokens)
            if block.kind is ChunkKind.TABLE
            else _pack_sentences(
                split_sentences(block.text),
                settings.chunk_target_tokens,
                settings.chunk_overlap_tokens,
            )
        )

        for piece in pieces:
            if not piece.strip() or estimate_tokens(piece) < _MIN_CHUNK_TOKENS:
                continue
            ordinal = per_section.get(section_key, 0)
            per_section[section_key] = ordinal + 1
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(filing.filing_id, section_key, ordinal, piece),
                    filing_id=filing.filing_id,
                    text=piece,
                    kind=block.kind,
                    section=block.section,
                    company=filing.company,
                    ticker=filing.ticker,
                    fiscal_year=filing.fiscal_year,
                    source_url=filing.source_url,
                    ordinal=ordinal,
                    char_start=block.start,
                    char_end=block.end,
                    token_estimate=estimate_tokens(piece),
                )
            )

    log.info(
        "chunked_filing",
        filing=filing.filing_id,
        blocks=len(blocks),
        chunks=len(chunks),
        tables=sum(1 for c in chunks if c.kind is ChunkKind.TABLE),
    )
    return chunks
