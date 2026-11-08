# Technical Decisions

Confirmed technology choices, why they were chosen, and constraints. **No stack changes are
made without explicit agreement.** Open questions are listed at the end and resolved inside
the phase that needs them.

## Confirmed Decisions

| # | Decision | Rationale / Notes | Phase |
| --- | --- | --- | --- |
| 1 | **Python 3.12** | Runtime baseline for all backend work. | 1 |
| 2 | **FastAPI + Pydantic** | Async API, typed validation, native SSE support for streaming answers. | 1, 5 |
| 3 | **Custom controlled scraper (httpx + BeautifulSoup)** | Full control of politeness (robots.txt, rate limits, allowlists) and scope. LangChain is used for Documents/splitting **only** — not as a crawler. | 2 |
| 4 | **LangChain Document abstraction + Text Splitters** | Standard, composable doc/chunk representation shared across ingestion, retrieval and RAG. | 2 |
| 5 | **`intfloat/multilingual-e5-small` embeddings** | Small, fast, multilingual. **Constraint:** E5 models require `query: ` / `passage: ` prefixes (handled centrally in `embeddings` module); 384-dim vectors. | 3 |
| 6 | **Qdrant + LangChain Qdrant integration** | VectorDB of record; collection + payload schema is the key design artifact of Phase 3. | 3 |
| 7 | **PostgreSQL + SQLAlchemy** | Conversations, messages, application/document metadata; also backing store for dedup registry. | 1 |
| 8 | **Hybrid retrieval: BM25 + vector, fused with RRF** | Lexical recall for API names/exact identifiers complements semantic recall; RRF needs no scoring calibration. | 4 |
| 9 | **Cross-encoder reranking** | Rerank fused candidates for precision. Specific model = Open Question. | 4 |
| 10 | **Query rewriting** | Standalone-query rewriting from conversation history before retrieval. | 4 |
| 11 | **LiteLLM with Groq first** | Single uniform interface for chat completion; provider swappable without touching RAG code. Groq model selection happens during implementation (Open Question). | 5 |
| 12 | **LangGraph for CRAG** | Explicit state machine for retrieval validation, corrective loops, fallback and abstention. | 6 |
| 13 | **Langfuse** | Traces, latency and token/cost usage across both pipelines, wired from Phase 5 onward. | 5+ |
| 14 | **RAGAS + Promptfoo** | RAGAS for retrieval/answer quality metrics; Promptfoo for prompt variant iteration; shared golden dataset. | 8 |
| 15 | **React + TypeScript** | Frontend framework; scaffolded with Vite in Phase 1. | 1, 7 |
| 16 | **Docker + Docker Compose, GitHub Actions, Pytest** | Local orchestration, CI/CD, testing. Kubernetes intentionally out of scope. | 1, 9 |
| 17 | **Git** | Version control; repo initialized at start of Phase 1. | 1 |
| 18 | **uv (backend dependency manager)** | **Resolved 2026-11-08** — was Open Question "uv vs Poetry". uv provides `pyproject.toml`-based dependency resolution with a lockfile (`uv.lock`), built-in Python version pinning (`.python-version`), and fast installs. | 1 |
| 19 | **SQLite document registry (stdlib) for ingestion dedup** | URL-keyed registry + content hashes; idempotent re-runs without PostgreSQL. Isolated in `deduplication.py` so it can be replaced in Phase 3+ if needed. | 2 |
| 20 | **URL normalization drops fragments AND query strings** | Doc pages are stateless; avoids per-tracking-parameter duplicate crawling. Path-aware scheme/host lowercase + `www.`-stripping + trailing-slash collapse. Boundary-aware domain allowlist matching (`qdrant.tech` ≠ `notevildomain-*`). | 2 |
| 21 | **Robots.txt via stdlib `urllib.robotparser`; unreachable robots = permissive (logged)** | No extra dependency; "respect where applicable" without making downtime fatal. Verified that a rule-less parser may deny-by-default — we mark `allow_all` explicitly on missing/failed fetch. | 2 |
| 22 | **Section-per-Document + heading-path prefix in chunk text** | One `Document` per H1/H2/H3-bounded section; `heading_hierarchy` metadata AND a text prefix (`H1 > H2 > H3`) so section context survives splitting and later retrieval. | 2 |
| 23 | **Single chunking algorithm: `RecursiveCharacterTextSplitter` (1000 / 150; `\n\n` → `\n` → `". "` → `" "`)** | Exactly one strategy per task constraints; params are the initial configuration, re-tuned later via the evaluation phase. Code blocks are fenced to maximize intactness. No MarkdownHeaderTextSplitter / semantic chunking. | 2 |
| 24 | **CLI: stdlib `argparse`, module entry `python -m app.ingestion.cli`** | No extra CLI dependency (Typer/click unnecessary for one command + options). | 2 |

## Deliberately Out of Scope

- **Kubernetes** — unnecessary complexity at this stage; Compose suffices.
- **Neo4j / knowledge graph RAG** — not part of the confirmed architecture.
- **Managed cloud services** — local-first, self-hosted dependencies.

## Open Questions (to resolve within the relevant phase)

| Question | Context | Phase |
| --- | --- | --- |
| Dependency manager | **Resolved → uv** (see Confirmed Decisions #18). | 1 |
| **Alembic** for SQLAlchemy migrations vs. bootstrap-only schema | Migration flow needs to be defined for Postgres. | 1 |
| **BM25 library**: `rank-bm25` vs. a search-engine-backed index | Drives BM25 persistence format and scale limits. | 2–4 |
| **Cross-encoder model**: e.g. `BAAI/bge-reranker-v2-m3` (multilingual, E5-compatible ecosystem) vs. smaller variants | Needs benchmarking against our docs; must be quantified for latency. | 4 |
| **PDF loader**: `pypdf`-based LangChain loader vs. higher-level option | Phase 2 shipped without PDFs (excluded from scope); decision moves to whenever PDF support starts. | deferred |
| **Groq model**: e.g. `llama-3.3-70b-versatile` (placeholder in `.env.example`) | Balanced against latency, context window, citation following. | 5 |
| BM25 **incremental update** strategy when new documents are indexed | Depends on chosen library; may require periodic rebuilds. | 3–4 |
