"""Pydantic schemas for the Chat API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# --- Chat ---

class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="Role of the speaker: 'user' or 'assistant'")
    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("role")
    def role_must_be_user_or_assistant(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str = Field(..., min_length=1, description="User's message")
    conversation_id: Optional[str] = Field(
        None, description="Existing conversation ID; create new if null"
    )

    @field_validator("message")
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("message must not be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    conversation_id: str = Field(..., description="Conversation identifier")
    answer: str = Field(..., description="Generated answer text")
    citations: List[dict] = Field(
        default_factory=list, description="Source citations"
    )


# --- Conversations ---

class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""
    id: str = Field(..., description="Conversation UUID")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    message_count: int = Field(
        default=0, description="Number of messages in the conversation"
    )


class ConversationDetail(ConversationSummary):
    """Full conversation detail with messages."""
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Conversation messages"
    )


class ConversationCreate(BaseModel):
    """Request to create a new conversation (implicit via chat)."""


class ConversationDelete(BaseModel):
    """Request to delete a conversation."""

    id: str = Field(..., description="Conversation UUID to delete")

    @field_validator("id")
    def id_must_be_valid(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("id must be a non-empty string")
        return v