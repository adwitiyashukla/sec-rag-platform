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

Measured on this project, from the installed virtual environment:

| Component | Installed size |
|---|---:|
| `onnxruntime` | 42.5 MB |
| `tokenizers` | 7.5 MB |
| `fastembed` | 0.8 MB |
| **ONNX inference stack, total** | **50.8 MB** |

For comparison, `torch` alone is between roughly 800 MB and 2.5 GB installed
depending on platform and whether CUDA is bundled, and `sentence-transformers`
pulls it in unconditionally. The difference is an order of magnitude, and it is
the difference between an image a free tier will build and one it will not.

Model weights are the same either way, and are also measured:

| Model | Size |
|---|---:|
| `BAAI/bge-small-en-v1.5` (dense) | 64.1 MB |
| `Xenova/ms-marco-MiniLM-L-6-v2` (reranker) | 87.5 MB |
| `prithivida/Splade_PP_en_v1` (learned sparse) | 508.1 MB |

That last row is why SPLADE ships behind a flag and is disabled by default in
the container: it is larger than the entire rest of the runtime combined.

Dense throughput measured at **307 documents per second** on CPU, which is not
the bottleneck anywhere in this pipeline.

What is given up: PyTorch's ecosystem. Fine-tuning an embedding model, or using
a model with no ONNX export, would require adding torch back. Since no model is
fine-tuned here, that cost is not currently paid.

A subtlety discovered while implementing this: `fastembed` does **not** apply
BGE's asymmetric query prefix in `query_embed`. Verified by comparing vectors,
`query_embed` and `embed` return identical output, and the manually prefixed
query differs at cosine 0.969. The prefix is therefore applied explicitly in
`Embedder.embed_query`. Omitting it is a silent quality regression rather than
an error.
