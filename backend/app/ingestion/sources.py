"""Documentation source definitions for the controlled scraper.

A :class:`DocSource` is the *only* way to tell the crawler what to visit:
allowed domains and (optionally) a path prefix act as hard boundaries, so the
scraper can never escape into unrelated parts of a site or other websites.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocSource:
    """Configuration for one documentation site to ingest."""

    name: str  # CLI identifier, e.g. "fastapi"
    display_name: str  # human-readable, e.g. "FastAPI"
    seed_url: str  # crawl entry point
    allowed_domains: tuple[str, ...]  # never leave these domains
    path_prefixes: tuple[str, ...] = ()  # empty = whole domain
    description: str = ""


DOC_SOURCES: tuple[DocSource, ...] = (
    DocSource(
        name="fastapi",
        display_name="FastAPI",
        seed_url="https://fastapi.tiangolo.com/",
        allowed_domains=("fastapi.tiangolo.com",),
        description="Official FastAPI documentation.",
    ),
    DocSource(
        name="qdrant",
        display_name="Qdrant",
        seed_url="https://qdrant.tech/documentation/",
        allowed_domains=("qdrant.tech",),
        path_prefixes=("/documentation",),
        description="Official Qdrant documentation (restricted to /documentation).",
    ),
    DocSource(
        name="docker-compose",
        display_name="Docker Compose",
        seed_url="https://docs.docker.com/compose/",
        allowed_domains=("docs.docker.com",),
        path_prefixes=("/compose",),
        description="Official Docker Compose docs (restricted to /compose).",
    ),
    DocSource(
        name="react",
        display_name="React",
        seed_url="https://react.dev/",
        allowed_domains=("react.dev",),
        description="Official React documentation. Note: react.dev is a client-rendered "
        "SPA, so quality of static extraction varies.",
    ),
    DocSource(
        name="postgresql",
        display_name="PostgreSQL",
        seed_url="https://www.postgresql.org/docs/current/",
        allowed_domains=("postgresql.org",),
        path_prefixes=("/docs/current",),
        description="Official PostgreSQL manual (restricted to /docs/current).",
    ),
)

_DOC_SOURCES_BY_NAME = {source.name: source for source in DOC_SOURCES}


def get_source(name: str) -> DocSource:
    """Look up a configured documentation source by CLI name."""
    try:
        return _DOC_SOURCES_BY_NAME[name]
    except KeyError:
        available = ", ".join(sorted(_DOC_SOURCES_BY_NAME))
        raise ValueError(
            f"Unknown documentation source {name!r}. Available: {available}"
        ) from None
