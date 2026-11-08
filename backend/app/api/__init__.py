"""API layer: FastAPI routers grouped by responsibility.

Each module in this package exposes an ``APIRouter`` which is assembled in
``app.main.create_app``.
"""

from app.api.health import router as health_router

__all__ = ["health_router"]
