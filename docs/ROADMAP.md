# Implementation Roadmap

Nine phases. Each phase lists its tasks, key deliverables and **exit criteria** — a phase is
"done" only when its exit criteria are met. Phases are sequential; dependencies are noted.

## Phase 1 — Project Foundation

**Tasks**
- Dependency manager + `pyproject.toml` (resolved: **uv** — see docs/TECHNICAL_DECISIONS.md Confirmed Decisions #18)
- FastAPI app skeleton (app factory, health endpoint, settings module, logging)
- React + TypeScript scaffold (Vite)
- Docker Compose: PostgreSQL, Qdrant (+ optional Langfuse), backend & dev containers
- SQLAlchemy setup: engine, session, initial models (conversations, messages, documents)
- `.env.example` refinement, README quickstart

**Deliverables**: `docker compose up` boots a working local stack; `/health` returns 200;
frontend dev server renders a placeholder page and can call the backend.

**Exit criteria**: full local stack starts with one command against `.env`; lint + placeholder
tests pass. *(Depends on: nothing.)*

## Phase 2 — Document Ingestion

**Status: implemented (2026-11-08).** Exit criteria verified: re-running ingestion yields
no duplicates (registry filter + content-hash skip) and each page leaves a raw-HTML +
sidecar audit trail under `data/raw/<source>/`. Note: the PDF loader listed below was
explicitly excluded from the Phase 2 implementation scope and is deferred (task decision;
tracked for a later phase).

**Tasks**
- ✅ Controlled scraper: robots.txt respect, per-domain allowlist + path prefixes, depth/page limits, rate limiting, resume via registry
- ✅ HTML parsing & cleaning: main-content extraction, boilerplate stripping
- ✅ Code-block preservation in cleaned text (fenced blocks + language detection)
- ➖ PDF loader — deferred (out of Phase 2 scope)
- ✅ LangChain Documents + metadata schema (URL, title, heading path, source name, type)
- ✅ RecursiveCharacterTextSplitter configured for developer docs (1000/150, ordered separators)
- ✅ Deduplication: URL normalization + content hashing (SQLite doc registry — stdlib, no PostgreSQL yet)
- ✅ CLI to ingest the five initial doc sources (`python -m app.ingestion.cli`)

**Deliverables**: cleaned, chunked, deduplicated documents from the sources in
`data/processed/<source>/chunks.jsonl`. Verified live with a limited FastAPI crawl
(5 pages → 97 section documents → 126 chunks; re-run fetched 0 pages).

**Exit criteria**: re-running ingestion produces no duplicates; scraping audit log (URLs,
robots decisions) exists. *(Depends on: Phase 1.)*

## Phase 3 — Embeddings & Qdrant

**Tasks**
- Sentence-Transformers wrapper with correct E5 `passage:` / `query:` prefixes
- Qdrant collection setup (dimension, distance metric, payload schema)
- LangChain Qdrant integration as the vector store layer
- Batched, resumable, **idempotent** indexing (content-hash upserts)
- Simple reindex/clear-dataset commands
- Metadata payload design for filtering by source/doc type (needed for PDF vs. web docs)

**Deliverables**: populated Qdrant collection (all five sources); indexing CLI.

**Exit criteria**: re-indexing a dataset yields an unchanged collection (no duplicates); representative semantic searches return topically correct chunks. *(Depends on: Phase 2.)*

## Phase 4 — Advanced Retrieval

**Tasks**
- Semantic search over Qdrant with metadata filtering
- BM25 index build + persistence (`data/indexes/`) — library to be chosen (see Open Questions)
- Hybrid search + RRF fusion
- Cross-encoder reranking — model to be chosen (see Open Questions)
- Query rewriting from conversation history
- Retrieval as a reusable interface so Phase 5/6 can call one abstraction

**Deliverables**: hybrid + RRF + reranker retrieval API (internal), benchmarked vs. pure vector search.

**Exit criteria**: measured retrieval-quality improvement (recall/precision on sample queries) vs. baseline; latency measured and acceptable. *(Depends on: Phase 3.)*

## Phase 5 — RAG Generation

**Tasks**
- LiteLLM integration (Groq) behind a thin provider wrapper
- Prompt templates: grounded-answer system prompt, citation format rules
- LangChain chains wired to the Phase 4 retrieval interface
- Citation extraction (URL + heading) demanded and enforced in outputs
- Conversation context handling; conversation/message persistence in PostgreSQL
- Streaming (SSE) endpoint + basic Langfuse tracing from day one

**Deliverables**: working `/chat` (REST + SSE) producing cited answers end-to-end.

**Exit criteria**: scripted evaluation shows cited answers; unanswerable questions don't produce fabricated sources. *(Depends on: Phase 4 + Phase 1 API.)*

## Phase 6 — Advanced RAG (LangGraph + CRAG)

**Tasks**
- LangGraph state machine over the query flow
- Retrieval validation node (relevance grading)
- CRAG corrective loops: rewrite + re-retrieve on weak retrieval
- Fallback & abstention paths
- LangGraph + Langfuse trace correlation

**Deliverables**: CRAG graph replacing the linear chain, with all transitions traced.

**Exit criteria**: eval shows fewer ungrounded answers than Phase 5 baseline; abstention behaves correctly. *(Depends on: Phase 5.)*

## Phase 7 — Frontend

**Tasks**
- React chat UI: conversation history, streaming responses, source citations (clickable)
- PDF upload UI + backend endpoints
- Conversation list / switch / delete (PostgreSQL-backed)
- Error, loading and empty states; basic theming

**Deliverables**: complete chat experience against Phase 5/6 endpoints.

**Exit criteria**: manual E2E walkthrough (new conversation → streaming answer with citations → follow-up → PDF upload + Q&A). *(Depends on: Phase 5/6.)*

## Phase 8 — Evaluation & LLMOps

**Tasks**
- Golden evaluation dataset (questions, expected answers, expected sources) in `evaluation/datasets/`
- RAGAS suites: retrieval quality (context recall/precision) + answer quality (faithfulness, answer relevancy)
- Promptfoo configs for prompt variant comparison
- Langfuse dashboards: latency, token usage, cost; alert thresholds
- Regression script runnable locally and in CI

**Deliverables**: reproducible eval reports; quality/latency/cost baselines recorded.

**Exit criteria**: one command reproduces the eval suite; metrics observable in Langfuse. *(Depends on: Phase 6.)*

## Phase 9 — DevOps & Production Readiness

**Tasks**
- Slim, multi-stage Docker images; health checks; Compose for prod-like profile
- GitHub Actions: lint, type check, Pytest, frontend build, image builds, eval smoke run
- Security: secret hygiene, dependency audit, upload validation & limits, rate limiting
- Deployment runbook

**Deliverables**: green CI/CD pipeline; documented deployment procedure.

**Exit criteria**: automated pipeline gates merges; deployment runbook tested once. *(Depends on: Phases 1–8.)*
