"""Application configuration loaded from environment variables via Pydantic Settings.

All settings mirror the keys documented in ``.env.example`` at the repository root.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (backend/app/core/config.py -> parents[3] == repo root).
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed application settings.

    Values are read from environment variables (case-insensitive), with an
    optional ``.env`` file acting as a fallback.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore keys for features implemented in later phases
    )

    app_name: str = "Smart Knowledge Assistant"
    app_version: str = "0.1.0"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # --- Controlled scraper (Phase 2) ---
    scrape_user_agent: str = "RAGAssistantBot/0.1 (+https://github.com/example/smart-knowledge-assistant)"
    scrape_max_depth: int = 3
    scrape_max_pages: int = 500
    scrape_request_delay_seconds: float = 1.0

    # --- Embeddings (Phase 3) ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 32

    # --- Qdrant (Phase 3) ---
    # "local" uses an embedded on-disk store (no service needed); "server" uses
    # the Qdrant instance at QDRANT_URL.
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_collection: str = "docs_chunks"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_local_path: Path = _REPOSITORY_ROOT / "data" / "qdrant"

    # Runtime data root (raw HTML, processed chunks, SQLite registry).
    data_dir: Path = _REPOSITORY_ROOT / "data"

    @property
    def debug(self) -> bool:
        """Convenience flag: in-development behaviour follows APP_ENV."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton (usable as a FastAPI dependency)."""
    return Settings()
