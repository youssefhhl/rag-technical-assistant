"""Application configuration loaded from environment variables via Pydantic Settings.

All settings mirror the keys documented in ``.env.example`` at the repository root.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (backend/app/core/config.py -> parents[3] == repo root).
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed application settings.

    Values are read from environment variables (case-insensitive), with an
    optional ``.env`` file acting as a fallback.
    """

    model_config = SettingsConfigDict(
        # .env lives at the repository root, NOT next to the launched process;
        # this keeps settings identical no matter where commands are run from.
        env_file=_REPOSITORY_ROOT / ".env",
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
    # Qdrant Server (Docker Compose: infrastructure/docker/docker-compose.yml).
    qdrant_collection: str = "docs_chunks"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # --- Cross-Encoder reranker (Phase 5) ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # --- Conversation history (Phase 6) ---
    database_url: str = (
        "postgresql+psycopg://rag_user:change-me@localhost:5432/rag_assistant"
    )
    conversation_history_window: int = 10

    # --- Query rewriting via LiteLLM/Groq (Phase 6) ---
    llm_model: str = "groq/llama-3.3-70b-versatile"
    groq_api_key: str | None = None

    # --- RAGAS evaluation (Phase 9.1) ---
    # Judge model for the evaluation metrics; falls back to llm_model when
    # unset. Uses the same GROQ_API_KEY (Groq OpenAI-compatible endpoint).
    evaluator_model: str | None = None
    evaluator_temperature: float = 0.0

    # Runtime data root (raw HTML, processed chunks, SQLite registry).
    data_dir: Path = _REPOSITORY_ROOT / "data"

    @property
    def debug(self) -> bool:
        """Convenience flag: in-development behaviour follows APP_ENV."""
        return self.app_env == "development"

    @field_validator("qdrant_api_key", "groq_api_key", mode="before")
    @classmethod
    def _empty_api_key_is_none(cls, value: object) -> object:
        """Treat an empty env value as "not set" (``KEY=`` in a .env file)."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton (usable as a FastAPI dependency)."""
    return Settings()
