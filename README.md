# Smart Knowledge Assistant

An advanced Retrieval-Augmented Generation (RAG) chatbot for developers. It ingests official
technical documentation (FastAPI, Qdrant, Docker Compose, React, PostgreSQL), stores it in a
vector database, and answers developer questions with an LLM. Every answer is **grounded in
retrieved documentation** and includes **source citations**. Manually uploaded PDF documents
are supported as an additional ingestion source.

> **Status:** Planning / foundation phase. Implementation has not started — see
> [docs/ROADMAP.md](docs/ROADMAP.md) for the phased plan.

## How It Works

The application is built around two pipelines (full detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

**A. Document Ingestion Pipeline**

```
Official doc sites → Custom controlled scraper (httpx + BeautifulSoup) → Raw HTML storage
→ Parsing & cleaning → LangChain Documents → Text Splitting → Chunks
→ HF Embeddings (multilingual-e5-small) → Qdrant indexing
```

**B. RAG Query Pipeline**

```
React chat → FastAPI → LangChain → Hybrid Retrieval (BM25 + Vector) → RRF
→ Cross-Encoder Reranking → Context → LLM (LiteLLM / Groq) → Answer + Citations
```

## Technology Stack

| Concern | Technology |
| --- | --- |
| Backend | Python 3.12, FastAPI, Pydantic |
| Ingestion | httpx, BeautifulSoup, custom controlled scraper, LangChain Documents + Text Splitters |
| RAG framework | LangChain, LangGraph (advanced workflows, e.g. Corrective RAG) |
| Embeddings | Hugging Face / Sentence Transformers — `intfloat/multilingual-e5-small` |
| Vector database | Qdrant + LangChain Qdrant integration |
| Relational database | PostgreSQL + SQLAlchemy |
| Retrieval | Semantic search, BM25, hybrid search, RRF, cross-encoder reranking |
| LLM integration | LiteLLM (initial provider: Groq) |
| Evaluation / LLMOps | RAGAS, Promptfoo, Langfuse |
| Frontend | React + TypeScript |
| DevOps | Docker, Docker Compose, Git, GitHub Actions, Pytest |

Decisions and their rationale are documented in
[docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md). No stack changes are made
without explicit agreement.

## Repository Layout

```
rag-assistant/
├── backend/            # FastAPI application (ingestion, RAG, API)
│   └── app/            # Organized by module responsibility (see below)
├── frontend/           # React + TypeScript chat interface (scaffolded in Phase 1)
├── docs/               # Architecture, roadmap, technical decisions
├── data/               # Runtime artifacts (not committed)
│   ├── raw/            # Raw HTML captured by the scraper
│   ├── processed/      # Cleaned / parsed documents
│   ├── pdf_uploads/    # User-uploaded PDFs
│   └── indexes/        # Persisted BM25 index
├── evaluation/         # RAGAS/Promptfoo artifacts
│   ├── datasets/       # Golden evaluation dataset
│   └── promptfoo/      # Promptfoo configs and prompts
└── infrastructure/     # Docker images and Docker Compose setup
    └── docker/
```

Backend modules (`backend/app/`):

| Module | Responsibility |
| --- | --- |
| `api/` | FastAPI routers, request/response schemas (chat, documents, health) |
| `core/` | Configuration, settings, logging |
| `database/` | SQLAlchemy models & sessions (conversations, messages, metadata) |
| `ingestion/` | Controlled scraper, HTML parsing/cleaning, PDF loading, splitting, dedup |
| `embeddings/` | Hugging Face sentence-transformers model wrapper |
| `vector_store/` | Qdrant client, collection management, indexing |
| `retrieval/` | Vector search, BM25, hybrid search + RRF, reranking, query rewriting |
| `rag/` | LLM via LiteLLM, prompt templates, LangGraph workflows (CRAG), citations |
| `evaluation/` | RAGAS / Promptfoo integration, quality and latency metrics |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipelines, data stores, module responsibilities
- [docs/ROADMAP.md](docs/ROADMAP.md) — 9-phase implementation plan with exit criteria
- [docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md) — confirmed stack, rationale, open questions

## Backend Quickstart (Phase 1 — implemented)

The FastAPI backend lives in [`backend/`](backend/) and uses **Python 3.12 + uv**.

```bash
cd backend

# Install dependencies (creates .venv and uv.lock, installs the app editable)
uv sync

# Run the tests (currently: app factory, configuration, /health endpoint)
uv run pytest

# Start the API (from the backend/ directory)
uv run uvicorn app.main:app --reload
```

Then open:

- Health check: <http://127.0.0.1:8000/health>
- Interactive docs: <http://127.0.0.1:8000/docs>

Configuration is loaded from environment variables (optionally from a `.env` file) via
Pydantic Settings — keys and defaults are documented in [`.env.example`](.env.example).
Useful keys today: `APP_ENV` (`development` | `staging` | `production`), `LOG_LEVEL`.

## Document Ingestion (Phase 2 — implemented)

A controlled scraper ingests official documentation sites into chunked LangChain
Documents, ready for the embedding phase (Phase 3).

```bash
cd backend

# List the five configured sources (fastapi, qdrant, docker-compose, react, postgresql)
uv run python -m app.ingestion.cli --list

# Ingest a source (limits controlled by SCRAPE_* settings in .env / .env.example)
uv run python -m app.ingestion.cli --source fastapi
# Per-run overrides:
uv run python -m app.ingestion.cli --source fastapi --max-pages 5 --depth 1 --delay 0.5
```

Outputs:

- Raw HTML + fetch metadata sidecars: `data/raw/<source>/`
- Chunks (JSONL, one JSON object per line with `page_content` + `metadata`):
  `data/processed/<source>/chunks.jsonl`
- Deduplication registry: `data/processed/ingestion_registry.sqlite3` — pages already
  ingested are skipped, so re-running a source is idempotent.

The scraper respects `robots.txt` (graceful fallback if unreachable), crawls only the
configured domain/path prefixes, and enforces max pages / depth / delay. Chunking uses a
single `RecursiveCharacterTextSplitter` (1000 chars / 150 overlap; separators `\n\n`,
`\n`, `. `, `' '`); each section Document carries its heading hierarchy
(`heading_hierarchy`) which is also prefixed into the chunk text.

> The full stack (frontend, PostgreSQL, Qdrant, Docker Compose) is implemented in later
> phases — see [docs/ROADMAP.md](docs/ROADMAP.md).
