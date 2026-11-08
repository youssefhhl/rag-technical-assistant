"""FastAPI application factory.

Entrypoints:
- ``create_app()``                     — application factory
- ``app`` (module-level instance)      — for ``uvicorn app.main:app``
"""

from fastapi import FastAPI

from app.api import health_router
from app.core.config import get_settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    """Build the FastAPI application instance.

    The factory keeps app construction testable and keeps configuration in
    ``app.core.config``.
    """
    setup_logging()
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "RAG chatbot for developer documentation. "
            "See docs/ARCHITECTURE.md for the target architecture."
        ),
    )

    # Endpoints are registered here as modules are implemented in later phases.
    application.include_router(health_router)

    return application


# Module-level instance so the app can be run without the ``--factory`` flag.
app = create_app()
