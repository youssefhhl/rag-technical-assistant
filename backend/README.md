# Backend

FastAPI application for the Smart Knowledge Assistant. Implemented in Phases 1–8
(see [../docs/ROADMAP.md](../docs/ROADMAP.md)).

## Layout

```
backend/
├── app/
│   ├── api/            # FastAPI routers, request/response schemas (chat, documents, health)
│   ├── core/           # Settings (Pydantic), logging, constants
│   ├── database/       # SQLAlchemy engine, session, ORM models
│   ├── ingestion/      # Controlled scraper, HTML parsers, PDF loader, splitting, dedup
│   ├── embeddings/     # Hugging Face embedding wrapper (E5 prefixes)
│   ├── vector_store/   # Qdrant client, collection management, batch indexing
│   ├── retrieval/      # Vector search, BM25, hybrid + RRF, reranking, query rewriting
│   ├── rag/            # LiteLLM integration, prompts, LangGraph CRAG, citations
│   └── evaluation/     # RAGAS runners, metrics
└── tests/              # Pytest suite
```

## Backend Quickstart

```bash
cd backend
uv sync                      # install dependencies (Python 3.12 + uv)
uv run pytest                # run the test suite
uv run uvicorn app.main:app --reload   # start the API at http://127.0.0.1:8000
```

Status (Phase 1): application factory, Pydantic Settings, `/health`, tests. Database,
retrieval and RAG modules are reserved for later phases (see
[../docs/ROADMAP.md](../docs/ROADMAP.md)).

Status (Phase 2): **ingestion is implemented** — see [Ingestion](#ingestion) below.

## Ingestion

Modules (in pipeline order — see `app/ingestion/pipeline.py` for the full flow):

| Module | Responsibility |
| --- | --- |
| `sources.py` | Configured doc sources: seed URL, allowed domains, path prefixes |
| `url_utils.py` | URL normalization (fragment/query dropped, `www.`/trailing-slash collapse), domain-allowlist & path checks, link extraction |
| `crawler.py` | Controlled BFS crawler (httpx): robots.txt, max pages/depth, request delay, visited-dedup, raw HTML + sidecar metadata storage under `data/raw/` |
| `parser.py` | BeautifulSoup extraction: main-content selection, boilerplate removal, ordered blocks (h1–h3, text, list items, code with language) |
| `preprocessor.py` | Tracks the H1/H2/H3 stack and groups blocks into `Section`s (heading hierarchy preserved) |
| `document_builder.py` | One LangChain `Document` per section: `source`, `title`, `source_name`, `document_type="website"`, `heading_hierarchy` metadata + heading path prefixed into `page_content` |
| `splitter.py` | `RecursiveCharacterTextSplitter` only: chunk 1000 / overlap 150, separators `\n\n` → `\n` → `". "` → `" "`; per-document `chunk_index` |
| `deduplication.py` | SQLite registry (`data/processed/ingestion_registry.sqlite3`) + content hashing; unchanged pages are skipped on re-runs |
| `pipeline.py` | Orchestration + in-run chunk dedup + JSONL persistence |
| `cli.py` | `uv run python -m app.ingestion.cli --list` / `--source <name> [--max-pages N] [--depth D] [--delay S]` |

Run the ingestion tests (fake site via `httpx.MockTransport`, no network):

```bash
uv run pytest tests/ingestion
```

## Intended dependency flow

```
api ─► rag ─► retrieval ─► vector_store ─► embeddings
                │                ▲
                └► ingestion ────┘        database ◄── (api, ingestion)
```

`core/` provides configuration shared by all; nothing else may import from `api/`.
