"""Tests for contextual query rewriting (LiteLLM/Groq fully mocked)."""

import sys
import types

import pytest
from pydantic import SecretStr  # noqa: F401  (kept for parity with settings users)

from app.core.config import Settings
from app.database.models import Message
from app.retrieval.query_rewriter import QueryRewriter


def make_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "scrape_request_delay_seconds": 0,
        "llm_model": "groq/llama-3.3-70b-versatile",
        "groq_api_key": None,
    }
    values.update(overrides)
    return Settings(**values)


class RecordingCompletion:
    """Fake LLM call; records prompt payloads, returns canned answers."""

    def __init__(self, response: str | Exception = "") -> None:
        self.calls: list[list[dict[str, str]]] = []
        self._response = response

    def __call__(self, messages) -> str:
        self.calls.append([dict(m) for m in messages])
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def history(*pairs):
    """Build duck-typed history messages quickly: pairs of (role, content)."""
    return [
        Message(id=i, conversation_id="c", role=role, content=content)
        for i, (role, content) in enumerate(pairs, start=1)
    ]


# -- prompt construction -------------------------------------------------------


def test_prompt_contains_system_rules_history_and_question():
    messages = QueryRewriter().build_prompt_messages(
        "how do I deploy it?",
        history(("user", "what is FastAPI?"), ("assistant", "A web framework.")),
    )

    system = messages[0]
    user = messages[1]
    assert system["role"] == "system" and "search query" in system["content"]
    assert user["role"] == "user"
    assert "User: what is FastAPI?" in user["content"]
    assert "Assistant: A web framework." in user["content"]
    assert "how do I deploy it?" in user["content"]


# -- rewriting behaviour ------------------------------------------------------


def test_rewrites_context_dependent_question_with_history():
    completion = RecordingCompletion("how do I deploy a FastAPI app with docker?")
    rewriter = QueryRewriter(settings=make_settings(), completion=completion)

    rewritten = rewriter.rewrite(
        "and deployment?", history(("user", "what is FastAPI?"))
    )

    assert rewritten == "how do I deploy a FastAPI app with docker?"
    assert len(completion.calls) == 1  # the LLM was consulted


def test_preserves_standalone_question_when_llm_returns_it_unchanged():
    completion = RecordingCompletion("how do I declare routes in FastAPI?")
    rewriter = QueryRewriter(settings=make_settings(), completion=completion)

    rewritten = rewriter.rewrite(
        "how do I declare routes in FastAPI?",
        history(("user", "hi")),
    )

    assert rewritten == "how do I declare routes in FastAPI?"


def test_empty_history_skips_the_llm_call():
    completion = RecordingCompletion("never")
    rewriter = QueryRewriter(settings=make_settings(), completion=completion)

    assert rewriter.rewrite("how are routes defined?", []) == "how are routes defined?"
    assert completion.calls == []  # no LLM call without history


def test_blank_question_is_returned_without_llm_call():
    completion = RecordingCompletion("never")
    rewriter = QueryRewriter(settings=make_settings(), completion=completion)

    assert rewriter.rewrite("   ", history(("user", "hi"))) == "   "
    assert completion.calls == []


def test_llm_failure_falls_back_to_original_question():
    completion = RecordingCompletion(RuntimeError("groq is down"))
    rewriter = QueryRewriter(settings=make_settings(), completion=completion)

    assert rewriter.rewrite("and deployment?", history(("user", "hi"))) == "and deployment?"


def test_empty_llm_answer_falls_back_to_original_question():
    completion = RecordingCompletion("")
    rewriter = QueryRewriter(settings=make_settings(), completion=completion)

    assert rewriter.rewrite("and deployment?", history(("user", "hi"))) == "and deployment?"


# -- real LiteLLM call path (module faked via sys.modules) ------------------------------------------------------


class _FakeChoiceContent:
    class message:  # noqa: N801 - mimics openai response shape
        content = "deploy it with docker compose"


class _FakeResponse:
    choices = [_FakeChoiceContent]


def test_call_llm_uses_litellm_completions_with_settings(monkeypatch):
    captured = {}

    fake_litellm = types.ModuleType("litellm")

    def fake_completion(model, messages, api_key=None, temperature=None, num_retries=None):
        captured.update(
            model=model, messages=messages, api_key=api_key, temperature=temperature
        )
        return _FakeResponse()

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    settings = make_settings(llm_model="groq/other", groq_api_key="gsk-test")
    rewriter = QueryRewriter(settings=settings)

    result = rewriter.rewrite("and deployment?", history(("user", "hi")))

    assert result == "deploy it with docker compose"
    assert captured["model"] == "groq/other"
    assert captured["api_key"] == "gsk-test"
    assert captured["temperature"] == 0.0


def test_real_litellm_mode_falls_back_when_provider_errors(monkeypatch):
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    rewriter = QueryRewriter(settings=make_settings())

    assert rewriter.rewrite("and deployment?", history(("user", "hi"))) == "and deployment?"
