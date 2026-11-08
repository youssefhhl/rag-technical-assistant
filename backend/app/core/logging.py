"""Logging configuration for the backend application."""

import logging

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure root logging using the ``LOG_LEVEL`` setting."""
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        force=True,  # reset any handlers configured elsewhere (e.g. uvicorn defaults)
    )
