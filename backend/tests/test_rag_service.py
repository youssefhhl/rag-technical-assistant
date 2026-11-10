"""Tests for RAG generation (Phase 7) — prompt building and answer parsing.

All LiteLLM calls are mocked; no real API calls are made.
"""

from app.core.config import Settings
from app.retrieval.rag_service import RAGAnswer, RAGService, RAG_PROMPT_TEMPLATE
from app.retrieval.hybrid_search import HybridHit
from app.retrieval.context_builder import ContextBuilder, ContextEntry


def make_rag_service(model: str | None = None, temperature: float | None = None) -> RAGService:
    from app.core.config import Settings
    kwargs: dict[str, object] = {"settings": Settings(_env_file=None)}
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    return RAGService(**kwargs)


def hit_from(text: str, metadata: dict | None = None) -> HybridHit:
    return HybridHit(
        content=text,
        metadata=metadata or {"title": "Doc", "source": "https://example.com", "chunk_index": 0},
        rrf_score=0.5,
        vector_rank=1,
        bm25_rank=1,
    )


# -- prompt template -----------------------------------------------------------

def test_prompt_template_has_required_sections():
    """The RAG prompt template must include context, question, and answer sections."""
    prompt = RAG_PROMPT_TEMPLATE.format(context="[chunk 1] ...", question="What is X?")
    assert "Context:" in prompt
    assert "User question:" in prompt
    assert "Answer:" in prompt


def test_prompt_includes_context_and_question():
    """The prompt must contain the actual context and question text."""
    prompt = RAG_PROMPT_TEMPLATE.format(
        context="[chunk 1] Paris is the capital of France.",
        question="What is the capital of France?",
    )
    assert "Paris" in prompt
    assert "What is the capital of France?" in prompt


def test_prompt_blank_context():
    """An empty context should still produce a valid prompt template."""
    prompt = RAG_PROMPT_TEMPLATE.format(context="", question="question?")
    assert "Context:" in prompt
    assert "User question:" in prompt
    assert "Answer:" in prompt


# -- RAGService -------------------------------------------------------------


def test_rag_service_prompt_building(monkeypatch):
    """The RAGService should build a prompt from hits and question."""
    rag = make_rag_service()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "Answer: Paris is the capital of France.")
    hits = [hit_from("Paris is the capital of France.")]
    answer = rag.generate("What is the capital of France?", hits)
    assert answer.answer.strip()


def test_rag_service_insufficient_info_mocked():
    """When the mocked LLM returns an insufficient-info response, the answer
    must reflect that."""
    rag = make_rag_service()
    # Monkey-patch _call_llm to return the insufficient-info string.
    rag._call_llm = lambda prompt: "I don't have enough information to answer this question based on the provided documents."
    hits = [hit_from("irrelevant text")]
    answer = rag.generate("What is the capital of France?", hits)
    assert "I don't have enough information" in answer.answer


def test_rag_service_preserves_citations(monkeypatch):
    """The RAG answer must include citations with title, source_url, chunk_index."""
    rag = make_rag_service()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "The answer is based on the provided context.")
    hits = [
        hit_from("context A", {"title": "Report A", "source": "urlA", "chunk_index": 1}),
        hit_from("context B", {"title": "Report B", "source": "urlB", "chunk_index": 2}),
    ]
    answer = rag.generate("question?", hits)
    assert len(answer.citations) == 2
    assert answer.citations[0]["title"] == "Report A"
    assert answer.citations[0]["source_url"] == "urlA"
    assert answer.citations[0]["chunk_index"] == "1"
    assert answer.citations[1]["title"] == "Report B"
    assert answer.citations[1]["source_url"] == "urlB"
    assert answer.citations[1]["chunk_index"] == "2"


def test_rag_service_empty_hits(monkeypatch):
    """With no hits, the answer must indicate insufficient information."""
    rag = make_rag_service()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "I don't have enough information to answer this question based on the provided documents.")
    answer = rag.generate("question?", [])
    assert "I don't have enough information" in answer.answer


def test_rag_answer_citations_are_valid(monkeypatch):
    """Each citation must have title, source_url, and chunk_index."""
    rag = make_rag_service()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "The capital of France is Paris.")
    hits = [hit_from("test content", {"title": "MyDoc", "source": "https://docs.example/test", "chunk_index": 3})]
    answer = rag.generate("question?", hits)
    assert isinstance(answer.answer, str)
    assert isinstance(answer.citations, list)
    assert len(answer.citations) == 1
    c = answer.citations[0]
    assert "title" in c
    assert "source_url" in c
    assert "chunk_index" in c
    assert c["chunk_index"] == "3"


def test_rag_answer_has_sufficient_length(monkeypatch):
    """A non-trivial answer should have more than 20 characters."""
    rag = make_rag_service()
    monkeypatch.setattr(rag, "_call_llm", lambda prompt: "The E5 model uses passage: prefix for queries and queries: prefix for query encoding.")
    hits = [hit_from("The E5 model uses passage: prefix for queries and queries: prefix for query encoding.")]
    answer = rag.generate("What prefix does E5 use for queries?", hits)
    assert len(answer.answer) > 20


def test_rag_service_with_custom_model():
    """The RAGService should respect a custom model override."""
    rag = make_rag_service(model="custom/model")
    assert rag.model == "custom/model"


def test_rag_service_custom_temperature():
    """The RAGService should respect a custom temperature."""
    rag = make_rag_service(temperature=0.5)
    assert rag.temperature == 0.5
