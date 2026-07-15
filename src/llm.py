"""Local grounded generation with Flan-T5 (seq2seq, CPU-friendly, no API key).

The prompt is deliberately strict: answer ONLY from the numbered context
passages, cite them by their [n] marker, and if the context is insufficient say
exactly INSUFFICIENT_CONTEXT. That sentinel is the model-side half of the
refusal mechanism (the retrieval-score threshold is the other half)."""
from __future__ import annotations

from .config import settings
from .schemas import ScoredChunk

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

_PROMPT = """You are a compliance assistant. Answer the QUESTION using ONLY the \
numbered CONTEXT passages below. Every claim must be supported by a passage and \
cited with its [number]. If the passages do not contain the answer, reply with \
exactly: {sentinel}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

_tok = None
_gen = None


def _load():
    global _tok, _gen
    if _gen is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(settings.generator_model)
        _gen = AutoModelForSeq2SeqLM.from_pretrained(settings.generator_model)
    return _tok, _gen


def build_context(scored: list[ScoredChunk]) -> str:
    lines = []
    for i, sc in enumerate(scored, start=1):
        lines.append(f"[{i}] ({sc.chunk.citation()}) {sc.chunk.text}")
    return "\n\n".join(lines)


def generate(question: str, scored: list[ScoredChunk], max_new_tokens: int = 256) -> str:
    tok, gen = _load()
    prompt = _PROMPT.format(
        sentinel=INSUFFICIENT,
        context=build_context(scored),
        question=question,
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = gen.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    return tok.decode(output[0], skip_special_tokens=True).strip()
