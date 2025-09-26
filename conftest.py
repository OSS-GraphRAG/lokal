# tests/conftest.py
import pytest
from types import SimpleNamespace
from langchain_core.documents import Document

class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        # Return as list or ndarray-like; tests will coerce to np array in the code
        # Maintain len == len(pairs)
        if callable(self._scores):
            return self._scores(pairs)
        return self._scores[:len(pairs)]

class FakeLLMStructured:
    """Mimics llm.with_structured_output(...) .invoke(...) API, returning a SimpleNamespace with attributes."""
    def __init__(self, result_dict):
        self._result_dict = result_dict

    def invoke(self, _inputs):
        return SimpleNamespace(**self._result_dict)

class FakeLLMText:
    """Mimics .invoke(...) returning an object with .content."""
    def __init__(self, content):
        self._content = content

    def invoke(self, _inputs):
        return SimpleNamespace(content=self._content)

class FakeToolsRetriever:
    def __init__(self, items):
        self._items = items

    def invoke(self, _query):
        return self._items

@pytest.fixture
def sample_docs():
    return [
        Document(page_content="alpha about M&A"),
        Document(page_content="beta about real estate"),
        Document(page_content="gamma about audits"),
    ]

@pytest.fixture
def base_state(sample_docs):
    return {
        "original_question": "How do QSBS rules apply to my startup exit?",
        "question": "How do QSBS rules apply to my startup exit?",
        "chat_history": [],
        "practice_area": "Generic_Tax_Law",
        "documents": sample_docs,
        "documents_are_relevant": "no",
        "generation": None,
    }
