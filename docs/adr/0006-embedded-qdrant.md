# ADR 0006: Embedded Qdrant, with BM25 derived from it

Status: accepted

## Context

The system needs a vector store supporting named vectors, sparse vectors, and
metadata filtering. It must run on a free tier with no separate service, and
survive a container restart.

## Decision

Qdrant in embedded mode, with the index as a directory on disk. The BM25 index
is rebuilt from the vector store at startup rather than persisted separately.

## Consequences

No server, no container orchestration, no network hop. The store holds the full
chunk payload and is the single source of truth for the corpus.

Deriving BM25 from the store rather than persisting it removes an entire class
of bug where two indexes silently drift apart after a partial ingestion. The
rebuild costs well under a second at this corpus size. It would not scale to
millions of chunks, at which point BM25 belongs in the store as a sparse vector
too, which Qdrant already supports.

Filtering happens inside the store rather than after retrieval. Post-filtering a
top-k list leaves almost nothing when the filter is selective, because the k
slots were already spent on documents that are then discarded.

Two limitations of embedded mode worth recording: payload indexes are accepted
but ignored (the calls are kept so the schema is correct against a real server,
and the warning is suppressed), and only one process may hold the index lock, so
the API runs single-process. Both are acceptable at this scale and both are
reasons to move to a Qdrant server rather than to change the code.
