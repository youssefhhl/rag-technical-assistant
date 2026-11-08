"""Database layer.

Reserved for the SQLAlchemy engine, session management and ORM models
(conversations, messages, document metadata). The relational database stack
(PostgreSQL + SQLAlchemy) is deliberately **not** installed or configured yet:
that happens when database work starts in Phase 1 infrastructure steps, per
``docs/ROADMAP.md``. No application code imports from this module today.
"""
