# Architecture

The application has two main pipelines — **Document Ingestion** (A) and **RAG Query** (B) —
plus three persistence layers and cross-cutting observability.

## High-Level View

```
┌───────────────────────────────  Pipeline A: Ingestion  ──────────────────────────────┐
│                                                                                      │
│  Official doc sites ──► Controlled scraper ──► Raw HTML ──► Parse & clean ──►        │
│  (FastAPI, Qdrant,       (httpx + BS4)          (data/raw/)      (data/processed/)   │
│   Docker Compose,                                                                  │
│   React, PostgreSQL)                          ┌────────────────────────────────────┐
│                                               │ PDFs (manual upload)               │
│                                               └────────────────────────────────────┘
│                                                        │
│                              LangChain Documents + metadata ──► Text splitter ──►
│                                        chunks ──► HF embeddings ──► Qdrant
└──────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────  Pipeline B: Query  ─────────────────────────────────┐
│                                                                                     │
│  React chat ──► FastAPI ──► LangChain/LangGraph ──► Hybrid retrieval                │
│             (PostgreSQL: conversations & messages)   (Vector + BM25)                │
│                                                        │                            │
│                          RRF fusion ──► Cross-encoder reranking ──► Context         │
│                                                        │                            │
│                          LLM via LiteLLM (Groq) ──► Answer + Citations              │
│                                     (streamed back to React; Langfuse traces all)   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline A — Document Ingestion

**Status: implemented (Phase 2).** Modules live in `backend/app/ingestion/` —
`sources → url_utils → crawler → parser → preprocessor → document_builder →
splitter → deduplication → pipeline`, with a CLI (`python -m app.ingestion.cli`).

1. **Custom controlled scraper** (`crawler.py`): httpx for HTTP, BeautifulSoup for link
   extraction. Only configured `DocSource`s are crawlable (domain allowlists + per-source
   path prefixes), with robots.txt respect, max pages/depth limits, request delay and
   per-run visited-dedup. LangChain is **only** used for document abstraction and
   splitting — never as a crawler replacement.
2. **Raw HTML storage**: each fetched page is stored under `data/raw/<source>/<hash>.html`
   with a sidecar `<hash>.json` (URL, timestamp, hash) — the scraping audit log required
   by `docs/ROADMAP.md`.
3. **Parsing & cleaning** (`parser.py`): main-content selection (`main`, `article`,
   `[role=main]`, docs-framework fallbacks), boilerplate removal (nav, header, footer,
   scripts, styles, ...), ordered content blocks; code-block languages detected from
   `language-*` / `highlight-*` CSS classes, heading anchor markers (`¶`) stripped.
4. **Structure-aware preprocessing** (`preprocessor.py`): the H1/H2/H3 stack is tracked and
   every paragraph/list/code block is bound to its heading hierarchy — e.g.
   `Docker Compose > Services > Ports` (kept in section metadata, see below).
5. **LangChain Documents** (`document_builder.py`): one Document per section with metadata
   `source` (URL), `title`, `source_name`, `document_type="website"`, `heading_hierarchy`;
   the heading path is also prefixed into `page_content` so section context travels into
   the chunks.
6. **Text splitting** (`splitter.py`): a single `RecursiveCharacterTextSplitter` — chunk
   size 1000, overlap 150, separators `\n\n` → `\n` → `". "` → `" "`. Splitting runs per
   parent Document; each child gets a per-document `chunk_index`. Code blocks are fenced
   (` ``` `) and therefore kept intact up to the chunk-size limit.
7. **Deduplication** (`deduplication.py`): URL-normalized SQLite registry with content
   hashing. Pages already ingested are filtered *before* fetching (idempotent re-runs) and
   identical chunk texts are dropped in-run. No PostgreSQL, per Phase 2 scope.
8. **Chunk persistence**: `data/processed/<source>/chunks.jsonl` (one JSON object per
   line: `page_content` + `metadata`) — the hand-off format for the embedding phase.

**Detail decisions** (rationale in `docs/TECHNICAL_DECISIONS.md`): query strings and
fragments are dropped in URL normalization; unreachable robots.txt is treated as
permissive and logged; PDF loading is deferred (out of Phase 2 scope, see the roadmap).

**Embeddings** (next phase): `intfloat/multilingual-e5-small` via Sentence Transformers,
with the E5-mandated `passage: ` / `query: ` prefixes distinguishing indexed chunks from
queries, then **Qdrant indexing** of chunks with payload metadata, idempotent and batched
(Phase 3).

## Pipeline B — RAG Query

1. **React chat interface** (Phase 7): conversations, streaming responses, citations, PDF upload.
2. **FastAPI** (Phase 1/5): REST + streaming (SSE) endpoints; conversation CRUD backed by PostgreSQL.
3. **LangChain orchestration**: query processing, prompt assembly, LLM invocation.
4. **Hybrid retrieval** (Phase 4): semantic vector search (Qdrant) **plus** BM25 lexical search
   (persisted index in `data/indexes/`), fused with **Reciprocal Rank Fusion (RRF)**.
5. **Cross-encoder reranking** (Phase 4): fused candidates rescored by a cross-encoder model.
6. **Query rewriting** (Phase 4): conversation context (last turns) is used to rewrite
   follow-up questions into standalone queries before retrieval.
7. **LLM via LiteLLM**: provider-agnostic access, initially Groq. Answers must cite the
   retrieved sources (URL + heading).
8. **Conversation context**: message history stored in PostgreSQL and injected into prompts.

## Advanced RAG — LangGraph

From Phase 6, the linear query flow is orchestrated as a **LangGraph** state machine enabling
**Corrective RAG (CRAG)**:

- a **retrieval validation** step grades retrieved chunks for relevance;
- low-quality retrieval triggers corrective actions (query rewrite + re-retrieval / fallback);
- if no grounding is achievable, the system **abstains** and says so instead of hallucinating.

## Data Stores

| Store | Contains | Used by |
| --- | --- | --- |
| `data/raw/` | Raw HTML pages captured by the scraper | Ingestion pipeline |
| `data/processed/` | Cleaned documents (pre-chunking) | Ingestion pipeline |
| `data/indexes/` | Persisted BM25 index | Hybrid retrieval |
| Qdrant | Chunk vectors + payload metadata | Retrieval, ingestion |
| PostgreSQL | Conversations, messages, document metadata (ingestion state, dedup records) | API, ingestion |
| `evaluation/datasets/` | Golden evaluation dataset | RAGAS / Promptfoo |

## Backend Module Responsibilities

| Module (`backend/app/...`) | Owns |
| --- | --- |
| `api/` | Routers, request/response schemas, streaming endpoints |
| `core/` | Settings (Pydantic), logging, shared constants |
| `database/` | SQLAlchemy engine, session, ORM models |
| `ingestion/` | Controlled scraper, parsers, splitters, dedup registry, orchestration (PDF loader deferred) |
| `embeddings/` | Embedding model wrapper (E5 prefixes handled here) |
| `vector_store/` | Qdrant client, collection setup/rebuild, batch indexing |
| `retrieval/` | Vector search, BM25, hybrid + RRF, reranking, query rewriting |
| `rag/` | LiteLLM integration, prompts, CRAG graph, citations |
| `evaluation/` | RAGAS runners, metric plumbing, golden dataset helpers |

## Cross-Cutting Concerns

- **Observability**: Langfuse collects traces, latency and token usage across both pipelines.
- **Configuration**: a single typed settings object (Pydantic) loaded from environment
  variables, matching `.env.example`.
- **Evaluation loops**: RAGAS (retrieval + answer quality) and Promptfoo (prompt variants)
  run against a versioned golden dataset kept in `evaluation/datasets/`.
- **Testing**: Pytest with fixtures for each layer; retrieval and RAG paths tested against
  small in-memory/local-stub components.
