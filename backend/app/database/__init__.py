"""Database layer (Phase 6): conversations/messages (SQLAlchemy) + repository.

Tables are managed by ``init_database``; the connection URL comes from
``DATABASE_URL`` (see ``.env.example``). Retrieval code only needs
``ConversationRepository.recent_messages`` for contextual query rewriting.

Note: ``ConversationRepository.append_message``/``recent_messages`` return ORM
``Message`` objects; consumers must treat them as read-only.
"""

from app.database.models import Base, Conversation, Message, utcnow
from app.database.repository import (
    ConversationRepository,
    Role,
    create_engine_from_url,
    init_database,
)

__all__ = [
    "Base",
    "Conversation",
    "ConversationRepository",
    "Message",
    "Role",
    "create_engine_from_url",
    "init_database",
    "utcnow",
]
