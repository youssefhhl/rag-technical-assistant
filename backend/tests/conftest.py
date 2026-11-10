import pytest

from app.database.repository import ConversationRepository
from app.database.models import Conversation, Message
from sqlalchemy import create_engine


@pytest.fixture
def repo():
    """Provide a fresh SQLite repository per test."""
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    engine = create_engine(f"sqlite:///{tmp}/test.db")
    from app.database import init_database
    init_database(engine)
    return ConversationRepository(engine)


@pytest.fixture
def seed_history(repo):
    """Seed a conversation with some messages."""
    def _seed(n=5, conversation_id="conv-1"):
        for i in range(n):
            repo.append_message(conversation_id, "user", f"message {i}")
        return conversation_id
    return _seed
