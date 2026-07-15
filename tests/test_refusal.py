"""Refusal-gate tests using a fake retriever/generator so no models are needed."""
from src.rag.schemas import Chunk, ScoredChunk, Answer
from src.rag.pipeline import RAGPipeline, REFUSAL_MESSAGE
from src.rag import pipeline as pipeline_mod
from src.rag.config import settings


class FakeRetriever:
    def __init__(self, score):
        self._score = score

    def retrieve(self, question, top_k=None):
        if self._score is None:
            return []
        ch = Chunk("id", "EDD requires senior approval.", "AML.pdf", "AML Policy", "§4.2", 2)
        return [ScoredChunk(chunk=ch, score=self._score)]


def test_refuses_when_no_hits():
    p = RAGPipeline(retriever=FakeRetriever(None))
    ans = p.answer("unrelated question about the moon")
    assert ans.refused and ans.answer == REFUSAL_MESSAGE


def test_refuses_below_threshold():
    low = settings.refusal_threshold - 0.05
    p = RAGPipeline(retriever=FakeRetriever(low))
    ans = p.answer("weakly related question")
    assert ans.refused


def test_answers_above_threshold(monkeypatch):
    high = settings.refusal_threshold + 0.5
    monkeypatch.setattr(pipeline_mod, "generate", lambda q, s: "EDD needs senior approval [1].")
    p = RAGPipeline(retriever=FakeRetriever(high))
    ans = p.answer("What does EDD require?")
    assert not ans.refused
    assert ans.citations and ans.citations[0].section == "§4.2"


def test_gate2_model_insufficient(monkeypatch):
    high = settings.refusal_threshold + 0.5
    monkeypatch.setattr(pipeline_mod, "generate", lambda q, s: "INSUFFICIENT_CONTEXT")
    p = RAGPipeline(retriever=FakeRetriever(high))
    ans = p.answer("something not covered")
    assert ans.refused
