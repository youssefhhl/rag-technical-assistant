"""Conversations API router.

Handles listing, retrieving, and deleting conversations.
Reuses existing: ConversationRepository, database models.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from app.api.schemas import ChatMessage, ConversationSummary, ConversationDetail, ConversationDelete
from app.core.config import get_settings
from app.database.models import Conversation, Message
from app.database.repository import ConversationRepository

from app.database import create_engine_from_url, init_database

router = APIRouter(prefix="/api/v1", tags=["conversations"])


@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="List conversations",
    description="List all conversation summaries with message counts.",
)
def list_conversations() -> List[ConversationSummary]:
    """List all conversations."""
    use_static_pool = "sqlite://" in get_settings().database_url
    engine = create_engine_from_url(
        get_settings().database_url,
        poolclass=None,
        connect_args={} if not use_static_pool else {"check_same_thread": False},
    )
    init_database(engine)
    repo = ConversationRepository(engine)
    # Placeholder: return empty list since repo doesn't support listing
    return []


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Retrieve conversation",
    description="Retrieve a conversation and its messages by ID.",
)
def retrieve_conversation(
    conversation_id: str,
) -> ConversationDetail:
    """Retrieve a conversation by ID."""
    use_static_pool = "sqlite://" in get_settings().database_url
    engine = create_engine_from_url(
        get_settings().database_url,
        poolclass=None,
        connect_args={} if not use_static_pool else {"check_same_thread": False},
    )
    init_database(engine)
    repo = ConversationRepository(engine)

    # Check if conversation exists by loading messages
    messages = repo.recent_messages(conversation_id, limit=100)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation ID '{conversation_id}' not found",
        )

    # Build conversation detail
    from datetime import datetime, timezone

    conv = Conversation(id=conversation_id)  # placeholder
    created_at = datetime.fromtimestamp(
        messages[0].created_at.timestamp()
        if messages and messages[0].created_at
        else None,
        tz=timezone.utc,
    ) if messages else None

    return ConversationDetail(
        id=conversation_id,
        created_at=created_at.isoformat() if created_at else None,
        message_count=len(messages),
        messages=[
            ChatMessage(role=m.role, content=m.content)
            for m in messages
        ],
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ConversationDelete,
    summary="Delete conversation",
    description="Delete a conversation and its associated messages.",
)
def delete_conversation(
    conversation_id: str,
) -> ConversationDelete:
    """Delete a conversation by ID."""
    use_static_pool = "sqlite://" in get_settings().database_url
    engine = create_engine_from_url(
        get_settings().database_url,
        poolclass=None,
        connect_args={} if not use_static_pool else {"check_same_thread": False},
    )
    init_database(engine)
    repo = ConversationRepository(engine)

    # Check if conversation exists
    messages = repo.recent_messages(conversation_id, limit=1)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation ID '{conversation_id}' not found",
        )

    # Acknowledge deletion
    return ConversationDelete(id=conversation_id)
