# ADR 0001: Run embedding models on ONNX Runtime rather than PyTorch

Status: accepted

## Context

The pipeline needs three transformer models: a bi-encoder for dense retrieval
(BGE-small), a cross-encoder for reranking (MiniLM), and optionally a learned
sparse encoder (SPLADE). The obvious choice is `sentence-transformers`, which is
what most RAG tutorials use.

The deployment target is a free CPU tier with no GPU, limited memory, and a
container that is rebuilt on every push.

## Decision

Use `fastembed`, which runs the same model weights through ONNX Runtime.

## Consequences

Measured on this project:

| | sentence-transformers + torch | fastembed + ONNX |
|---|---|---|
| Install size | roughly 2.5 GB | roughly 400 MB |
| Container cold start | 45 to 90 s | about 5 s |
| Dense throughput (CPU) | comparable | 307 docs/s measured |

The size reduction is what makes free tier hosting practical at all. A 2.5 GB
image is slow to build in CI, slow to pull, and close to the limits of the
platforms this targets.

What is given up: PyTorch's ecosystem. Fine-tuning an embedding model, or using
a model with no ONNX export, would require adding torch back. Since no model is
fine-tuned here, that cost is not currently paid.

A subtlety discovered while implementing this: `fastembed` does **not** apply
BGE's asymmetric query prefix in `query_embed`. Verified by comparing vectors,
`query_embed` and `embed` return identical output, and the manually prefixed
query differs at cosine 0.969. The prefix is therefore applied explicitly in
`Embedder.embed_query`. Omitting it is a silent quality regression rather than
an error.
