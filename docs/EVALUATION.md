# RAGAS Evaluation (Phase 9.1)

Quality evaluation of the production RAG pipeline with four RAGAS metrics.
The evaluator runs the **real** pipeline (Qdrant dense + BM25 sparse → RRF →
Cross-Encoder reranking → LiteLLM/Groq generation) on a golden dataset
grounded in the currently indexed FastAPI documentation — nothing is mocked.

## What is measured

| Metric | Meaning |
|---|---|
| `faithfulness` | Is the generated answer supported by the retrieved contexts? (hallucination check) |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are relevant retrieved contexts ranked high? (uses the golden reference) |
| `context_recall` | Do the retrieved contexts contain the golden-reference facts? |

A handful of **unanswerable** questions (outside the indexed corpus) probe the
pipeline's abstention behaviour; they are recorded but never scored against a
reference.

RAGAS version: **0.2.15** (`ragas>=0.2.15,<0.3`), exact per-sample API
(`SingleTurnSample` + `metric.single_turn_ascore`). A small compatibility shim
(`app/evaluation/ragas_compat.py`) registers the module name
`langchain_community.chat_models.vertexai` that every ragas release imports at
module level but which does not exist anymore with `langchain-community>=0.4`;
it raises loudly if ever instantiated (the evaluation never uses VertexAI).

## Run it

```bash
# smoke run (development)
uv run python -m app.evaluation.evaluation_runner --limit 5

# full evaluation over the golden dataset
uv run python -m app.evaluation.evaluation_runner
```

Requires the live services (PostgreSQL, Qdrant at :6333, BM25 index) and a
configured LLM key. Reports land in
`backend/app/evaluation/reports/evaluation_<UTC timestamp>.json` (gitignored).

## Configure the evaluator model

The RAGAS judge reuses the production environment (`GROQ_API_KEY` from
`.env`; it is never hardcoded) and talks to Groq's OpenAI-compatible endpoint.
It defaults to `LLM_MODEL` and can be overridden:

- environment: `EVALUATOR_MODEL=groq/openai/gpt-oss-120b` (LiteLLM-style
  `groq/<model>` ids; only `groq/`-prefixed models are accepted and the
  prefix is stripped for the direct API call), temperature fixed by
  `EVALUATOR_TEMPERATURE` (default `0.0`);
- CLI: `--model groq/qwen/qwen3.8-27b`.

Practical note verified on this deployment: `openai/gpt-oss-120b` sometimes
emits unparseable verdict JSON (ragas records a NaN score; the evaluator
retries once and then reports the failure explicitly). `qwen/qwen3.8-27b`
produces clean judgments for all four metrics on this dataset.

Other CLI options: `--dataset <path>` (default
`backend/app/evaluation/datasets/golden_dataset.json`) and `--output <dir>`.

## Report contents

Each JSON report contains: timestamp, dataset name/counts, RAGAS version,
evaluator model, aggregate mean per metric (n evaluated / n failed), explicit
per-question scores, `failed_or_skipped` counts with exact error messages,
the abstention observations for unanswerable probes, and the performance
section (total + average query latency; estimated cost per query is
explicitly `None`/unavailable because the production LiteLLM call does not
expose reliable token-usage/pricing without touching production code).

## Structural notes

- Retrieved contexts (real pipeline output) and reference material (golden
  excerpts) travel in **strictly separate** RAGAS fields:
  `retrieved_contexts` vs `reference`/`reference_contexts`
  (`app/evaluation/samples.py`).
- Evaluation runs are **read-only** for the conversations database: they never
  append messages and use a throw-away evaluation conversation id, so user
  history is untouched (`app/evaluation/pipeline.py`).
- Samples are scored sequentially (no concurrent bursts against Groq).
- No failed sample is ever turned into a fabricated score: failures are
  recorded with their exact errors (`ScoreOutcome` in `evaluator.py`).

## Extending

Questions/references follow a single schema (`answerable: false` marks a
probe). Add entries to `datasets/golden_dataset.json` when new documentation
sources are ingested: answerable questions require a non-empty, manually
verified `reference` plus at least one `reference_context` excerpt from the
official docs; malformed entries fail dataset validation at load time.
