"""API layer: FastAPI routers grouped by responsibility.

Each module in this package exposes an ``APIRouter`` which is assembled in
``app.main.create_app``.
"""

from app.api.chat_router import router as chat_router
from app.api.conversations_router import router as conversations_router
from app.api.health import router as health_router

__all__ = ["health_router", "chat_router", "conversations_router"]
