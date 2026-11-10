"""Session/engine setup and the conversation history repository (Phase 6)."""

from __future__ import annotations

import logging
from typing import Literal

import uuid
from sqlalchemy import create_engine, desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.database.models import Base, Conversation, Message

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


def create_engine_from_url(url: str, **engine_kwargs) -> Engine:
    """Build the SQLAlchemy engine for the configured database URL."""
    if url.startswith("sqlite://") and ":memory:" in url:
        # keep the in-memory database alive across connections (tests)
        engine_kwargs.setdefault("poolclass", StaticPool)
        engine_kwargs.setdefault("connect_args", {"check_same_thread": False})
    return create_engine(url, **engine_kwargs)


def init_database(engine: Engine) -> None:
    """Create the tables if missing (idempotent)."""
    Base.metadata.create_all(engine)
    logger.info("Database tables ensured (driver=%s)", engine.dialect.name)


class ConversationRepository:
    """Append user/assistant messages and load recent history per conversation."""

    def __init__(self, engine_or_sessionfactory, settings: Settings | None = None) -> None:
        if isinstance(engine_or_sessionfactory, Engine):
            self._session_factory = sessionmaker(
                bind=engine_or_sessionfactory, expire_on_commit=False
            )
        else:
            self._session_factory = engine_or_sessionfactory
        config = settings or get_settings()
        self.default_window = max(1, config.conversation_history_window)

    # -- writes ---------------------------------------------------------------

    def append_message(self, conversation_id: str | None, role: Role, content: str) -> Message:
        """Persist a message; the conversation row is created on the fly."""
        with self._session_factory() as session, session.begin():
            conversation: Conversation | None = None
            if conversation_id is not None:
                conversation = session.get(Conversation, conversation_id)
                if conversation is None:
                    # Create conversation with the provided ID so look-ups work immediately
                    conversation = Conversation(id=conversation_id)
                    session.add(conversation)
            else:
                # Auto-create a new conversation with a generated UUID
                import uuid
                conversation = Conversation(id=str(uuid.uuid4()))
                session.add(conversation)
            message = Message(conversation_id=conversation.id, role=role, content=content)
            session.add(message)
            session.flush()
            session.refresh(message)
            return message
            message = Message(conversation_id=conversation.id, role=role, content=content)
            session.add(message)
            session.flush()
            session.refresh(message)
            return message

    # -- reads ----------------------------------------------------------------

    def recent_messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        """The most recent ``limit`` (default: settings window) messages, oldest first."""
        window = self.default_window if limit is None else max(1, limit)
        with self._session_factory() as session:
            # ids are monotonic: take the N newest messages, then restore order
            newest = (
                session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(desc(Message.id))
                    .limit(window)
                )
                .scalars()
                .all()
            )
            return list(reversed(newest))
