"""Chat API router.

Handles the ``/chat`` endpoint.  Reuses the existing services:
``ConversationRepository``, ``ContextualSearchService``, ``HybridSearchService``
and ``RAGService``.

The retrieval pipeline wired here is the real one::

    Qdrant (dense, E5)  \\
                          -> RRF fusion -> Cross-Encoder -> ContextBuilder -> Groq
    BM25 (sparse)       /

No engine is stubbed out and no citation metadata is invented: every hit keeps
the ``title`` / ``source`` / ``chunk_index`` values of the document it came from.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import ChatRequest, ChatResponse
from app.core.config import get_settings
from app.database import create_engine_from_url, init_database
from app.database.repository import ConversationRepository
from app.retrieval.bm25_search import Bm25SearchService
from app.retrieval.contextual_search import ContextualSearchService
from app.retrieval.hybrid_search import HybridSearchService, SearchEngine
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.rag_service import RAGAnswer, RAGService
from app.retrieval.reranker import CrossEncoderReranker
from app.vector_store.qdrant_service import build_vector_store
from app.vector_store.search import SemanticSearchService

router = APIRouter(prefix="/api/v1", tags=["chat"])


# -- dependency providers ------------------------------------------------------


def get_conversation_repo() -> ConversationRepository:
    use_static_pool = "sqlite://" in get_settings().database_url
    engine = create_engine_from_url(
        get_settings().database_url,
        poolclass=None,
        connect_args={} if not use_static_pool else {"check_same_thread": False},
    )
    init_database(engine)
    return ConversationRepository(engine)


def get_rag_service() -> RAGService:
    return RAGService()


def _make_rewriter() -> QueryRewriter:
    """The LLM-backed query rewriter (history-aware)."""
    return QueryRewriter(get_settings())


def _make_qdrant_engine() -> SemanticSearchService:
    """The real dense engine: Qdrant + E5 query embeddings.

    ``SemanticSearchService`` embeds the query with the ``query: `` prefix and
    returns hits whose ``metadata`` follows the flat retrieval contract
    (``title`` / ``source`` / ``chunk_index`` / ``heading_hierarchy``).
    """
    return SemanticSearchService(build_vector_store())


def _make_bm25_engine() -> Bm25SearchService:
    """The real sparse engine: the persisted BM25 index of the corpus."""
    from app.retrieval.bm25_indexing import load_bm25_index

    try:
        bundle = load_bm25_index()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "BM25 index is missing - run the `bm25_indexing` step before serving chat requests"
        ) from exc
    return Bm25SearchService(bundle)


def _make_reranker() -> CrossEncoderReranker:
    """The Phase 5 Cross-Encoder reranker (model name comes from settings)."""
    return CrossEncoderReranker()


def get_search_service(
    repo: ConversationRepository = Depends(get_conversation_repo),
) -> ContextualSearchService:
    """Contextual search over the real Qdrant + BM25 engines."""
    hybrid = HybridSearchService(
        vector_search=_make_qdrant_engine(),
        bm25_search=_make_bm25_engine(),
        reranker=_make_reranker(),
    )
    return ContextualSearchService(
        repository=repo,
        query_rewriter=_make_rewriter(),
        hybrid_service=hybrid,
        window=get_settings().conversation_history_window,
    )


# -- endpoint ------------------------------------------------------------------


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the RAG assistant",
    description=(
        "Send a message to the RAG assistant. If conversation_id is null, "
        "a new conversation is created. The user message is saved, contextual "
        "search and RAG generation are executed, and the assistant response "
        "is persisted."
    ),
)
def chat(
    request: ChatRequest,
    repo: ConversationRepository = Depends(get_conversation_repo),
    search_service: ContextualSearchService = Depends(get_search_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    # append_message handles conversation creation on the fly.
    user_msg = repo.append_message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.message,
    )

    # Execute contextual search (query rewriting + hybrid retrieval + rerank).
    try:
        outcome = search_service.search(
            user_msg.conversation_id, request.message, top_k=get_settings().reranker_top_k
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a 503, never swallowed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    # The retrieved hits already carry the real chunk text and the real
    # document metadata - hand them to the generator unchanged.
    rag_answer: RAGAnswer = rag_service.generate(request.message, outcome.hits)

    # Save the assistant message
    repo.append_message(
        conversation_id=user_msg.conversation_id,
        role="assistant",
        content=rag_answer.answer,
    )

    # Return response
    return ChatResponse(
        conversation_id=user_msg.conversation_id,
        answer=rag_answer.answer,
        citations=rag_answer.citations,
    )
