# Phase 10 — Advanced RAG & Reliability

The reliability layer sits **between** the retrieval and the generator. It
reuses the existing components (Cross-Encoder, hybrid search, query rewriter,
context builder) and adds four guarantees, without changing the retrieval or
generation algorithms and without touching the Chat API contract.

```
                 +-------------------------------------------+
Qdrant + BM25   |  Phase 10 reliability layer                |
      |          |                                           |
      v          |  1. validation   (Cross-Encoder scores)   |
RRF -> rerank -->|  2. CRAG         (1 corrective retrieval) |
      |          |  3. abstention   (insufficient context)   |
      v          |  4. citations    (verified sources)       |
   generator <--+-------------------------------------------+
```

## 1. Retrieval validation — `app/retrieval/validation.py`

The Phase 5 Cross-Encoder already scores every reranked chunk
(`RerankHit.rerank_score`). The validator turns those scores into a decision:

| label | condition (top score) | behaviour |
|---|---|---|
| `RELEVANT` | `>= retrieval_relevant_threshold` (2.0) | generate |
| `PARTIALLY_RELEVANT` | `>= retrieval_partial_threshold` (-3.0) | generate (weak context) |
| `IRRELEVANT` | below, or empty retrieval | abstain (after one corrective attempt) |

The thresholds are **calibrated on the real logits** of
`cross-encoder/ms-marco-MiniLM-L-6-v2` (relevant FastAPI pairs: +2 … +11,
unrelated pairs: −11 … 0) and are configurable:

```bash
RETRIEVAL_RELEVANT_THRESHOLD=2.0
RETRIEVAL_PARTIAL_THRESHOLD=-3.0
```

Measured on the indexed corpus (real retrieval, top-5 hits):

```
RELEVANT     top=+8.41  How do I install FastAPI in a project using uv?
RELEVANT     top=+7.74  What is FastAPI?
RELEVANT     top=+5.01  Which libraries does FastAPI stand on?
RELEVANT     top=+4.91  How fast is FastAPI compared to other Python frameworks?
IRRELEVANT   top=-11.18  What is the weather in Paris tomorrow?
IRRELEVANT   top=-11.10  How do I bake a sourdough bread at home?
```

Deliberate choice: hits **without** a Cross-Encoder score (a pure-RRF
configuration) are classified `PARTIALLY_RELEVANT` — the retrieval cannot be
judged, so the pipeline stays permissive and never abstains by mistake.

## 2. Intelligent abstention — `app/retrieval/reliability.py`

When the context is judged insufficient, the generator is **not called** at
all. The answer is the same fixed sentence the generator already used
(`ABSTENTION_MESSAGE`, identical to `ABSTENTION_PLACEHOLDER` of the
evaluation dataset), with **no citation**:

```
I don't have enough information to answer this question based on the provided documents.
```

Empty retrieval results are handled by the same path (empty → `IRRELEVANT`).
The API response keeps its exact shape: `answer` + `citations`.

## 3. Lightweight CRAG — `app/retrieval/corrective.py`

One corrective round at most, never more:

1. **reformulate** the question with the existing `QueryRewriter`
   (`reformulate()` — same information need, different wording);
2. **one** new hybrid search (Qdrant + BM25 + RRF + Cross-Encoder) with the
   reformulated query;
3. **re-assess** the new result with the same thresholds;
4. generate if the context is now sufficient, otherwise abstain.

Guarantees: the corrective attempt is skipped when the reformulation is
identical to the original question (the search would return the same
documents); the original question is never modified — only the retrieval
query is. `CRAG_CORRECTION_ENABLED=false` disables the correction entirely.

## 4. Citation verification — `app/retrieval/citations.py`

Every citation is checked against the documents that were **really
retrieved** for that question:

- the source metadata must exist (`source_url`);
- the title metadata must exist;
- the `(source_url, chunk_index)` pair must match a retrieved chunk.

Invalid citations are rejected (dropped from the response) with an explicit
reason — they are never rewritten into another source. The citation format is
unchanged: `{"title": ..., "source_url": ..., "chunk_index": ...}`.

## Files

| file | role |
|---|---|
| `app/retrieval/validation.py` | scores → `RELEVANT` / `PARTIALLY_RELEVANT` / `IRRELEVANT` |
| `app/retrieval/citations.py` | citation verification against the retrieved documents |
| `app/retrieval/corrective.py` | one corrective retrieval (CRAG) |
| `app/retrieval/reliability.py` | orchestration: validate → correct → abstain → generate → verify |
| `app/retrieval/query_rewriter.py` | new `reformulate()` (same injectable LLM call) |
| `app/retrieval/contextual_search.py` | public accessors (`hybrid_service`, `query_rewriter`) |
| `app/api/chat_router.py` | wiring of the layer into `/api/v1/chat` |
| `app/core/config.py` | `retrieval_relevant_threshold`, `retrieval_partial_threshold`, `crag_correction_enabled` |

## RAGAS comparison (Phase 9 baseline)

The evaluation runner can run the pipeline **through** the reliability layer
to compare it with the Phase 9 baseline. Nothing is launched automatically —
the flag is opt-in:

```bash
# Phase 9 baseline (unchanged behaviour)
uv run python -m app.evaluation.evaluation_runner --limit 5

# Phase 10 pipeline (validation + abstention + CRAG + citation verification)
uv run python -m app.evaluation.evaluation_runner --limit 5 --reliability
```

The Phase 10 report adds two optional fields per question
(`retrieval_label`, `corrective_used`); with the flag off the Phase 9 report
format is byte-for-byte unchanged.

Both runs must use the same generator model to be comparable
(`LLM_MODEL=groq/openai/gpt-oss-20b`) and the same judge model
(`--model groq/qwen/qwen3.8-27b`).

## Tests

```bash
cd backend && uv run pytest tests/test_retrieval_validation.py \
                      tests/test_citation_verification.py \
                      tests/test_corrective_retrieval.py \
                      tests/test_reliability_pipeline.py \
                      tests/test_reliability_api.py -q
```

All tests mock the Cross-Encoder, the hybrid search, the query rewriter and
the generator: **no Groq quota is consumed** by the unit suite.

## Limitations

- The thresholds are calibrated on one Cross-Encoder and one corpus; another
  reranker model needs its own calibration.
- The classification uses the **best** hit only: a strong chunk among weak
  ones is enough to answer, which is intended but can let a partly unrelated
  chunk through.
- The corrective attempt costs one extra hybrid search plus one LLM call
  (reformulation) — it only happens when the first retrieval is judged
  irrelevant.
- Citation verification compares the *metadata* of the citations with the
  retrieved chunks; inline citations written inside the answer text are not
  rewritten (the answer text itself is never altered).
- The corrective attempt relies on the reformulation **keeping the subject**
  of the question (the prompt enforces it explicitly): a reformulation that
  drifts to another topic would let the validator accept an off-topic but
  well-scored context. The retrieval score stays the last line of defence.
