# Deployment and key setup

The system runs fully without any API key, using the deterministic offline
provider. Everything below is optional and takes about ten minutes.

---

## 1. Add an LLM key so answers are model-written

Without a key, `EchoProvider` composes extractive answers from the retrieved
passages. Citations, groundedness, and refusals all work; the prose is just
stitched from source sentences rather than written.

### Groq (recommended: fastest, no credit card)

1. Go to <https://console.groq.com/keys> and sign in.
2. Create an API key and copy it. It begins with `gsk_`.
3. Put it in `.env` in the project root:

   ```bash
   SECRAG_GROQ_API_KEY=gsk_your_key_here
   ```

4. Verify:

   ```bash
   secrag query "What supply chain risks does Apple disclose?"
   ```

   The diagnostics block should show a real answer with citations. Check
   `secrag stats` and confirm `providers` lists `groq:llama-3.3-70b-versatile`
   rather than `echo:echo-1`.

Free tier at time of writing: 30 requests per minute, 14,400 per day.

### Google Gemini (fallback arm)

1. Go to <https://aistudio.google.com/apikey> and create a key.
2. Add to `.env`:

   ```bash
   SECRAG_GEMINI_API_KEY=your_key_here
   ```

With both configured, `SECRAG_LLM_PROVIDERS=groq,gemini` tries Groq first and
falls through to Gemini on a rate limit, which roughly doubles usable
throughput at zero cost.

---

## 2. Deploy the live demo to Hugging Face Spaces

Free, always on, and it rebuilds automatically on every push to `main`.

### Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Name it `sec-rag-platform`.
3. Choose **Docker** as the SDK, and the **blank** template.
4. Visibility: public.
5. Create.

### Create a token

1. <https://huggingface.co/settings/tokens>
2. New token, type **Write**. Copy it.

### Add the repository secrets

In the GitHub repo, go to Settings, Secrets and variables, Actions, and add:

| Secret | Value |
|---|---|
| `HF_TOKEN` | the write token you just created |
| `HF_USERNAME` | your Hugging Face username |
| `HF_SPACE` | `sec-rag-platform` |

### Add the Space's own secrets

In the Space, go to Settings, Variables and secrets, and add `SECRAG_GROQ_API_KEY`.
Space secrets are separate from GitHub secrets; the container reads this one at
runtime.

### Deploy

Push to `main`, or trigger the **Deploy to Hugging Face Spaces** workflow
manually from the Actions tab. The workflow only runs after CI and the
evaluation gate have both passed, so a regression cannot reach the live demo.

The Space builds the Dockerfile, which pre-bakes the ONNX weights. First build
takes five to ten minutes; afterwards the container starts in about five seconds.

### One thing to know about Spaces

The free tier gives no persistent disk. The container starts with an empty
index, and the UI will correctly report zero chunks and refuse to answer until
one exists. Two options:

- **Ingest on boot.** Set `SECRAG_INGEST_ON_START=true` and have the container
  build the index at startup. Costs a few minutes of cold start and re-fetches
  from EDGAR on every restart.
- **Ship the index in the image** (recommended). Add to the Dockerfile after the
  source copy:

  ```dockerfile
  RUN python -m secrag.cli ingest --rebuild && python -m secrag.cli train-router
  ```

  This bakes a ready index into the image. The build is slower, the runtime
  needs no network, and the demo works instantly on first visit.

---

## 3. Enable SPLADE (optional)

The third retrieval arm is off by default in the container because it adds
532 MB against BGE's 67 MB and meaningful memory pressure on a free tier.

```bash
SECRAG_ENABLE_SPLADE=true
```

Run `secrag benchmark` before and after to see what it buys you on your corpus.

---

## 4. Local Docker

```bash
docker compose up --build
```

Then open <http://localhost:7860>. The compose file mounts a named volume at
`/data`, so the index survives restarts.

---

## Troubleshooting

**EDGAR returns 403.** Your `SECRAG_EDGAR_USER_AGENT` is missing a contact
address. EDGAR requires a descriptive user agent; this is a documented
requirement, not an optional courtesy.

**`secrag query` refuses everything.** The index is empty. Run `secrag ingest`
and check `secrag stats` reports a non-zero `corpus_chunks`.

**Answers come back with `status: refused_ungrounded`.** The retrieved passages
did not support the generated claims. That is the guardrail working. Lower
`SECRAG_MIN_GROUNDEDNESS` to see the withheld answer, or check whether the
corpus actually contains the topic.

**The router routes everything to `factoid`.** The classifier is not trained.
Run `secrag train-router`.

**`ltr` reranker silently behaves like `none`.** The model is not trained. Run
`secrag train-ltr`. It degrades to fusion order by design rather than failing,
because an untrained ranker should never make results worse than no ranker.
