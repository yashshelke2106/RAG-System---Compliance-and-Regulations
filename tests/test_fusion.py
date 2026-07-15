from dataclasses import dataclass
from src.rag.hybrid_retriever import reciprocal_rank_fusion


@dataclass
class C:
    chunk_id: str


def _mk(ids):
    return [(C(i), 1.0) for i in ids]


def test_rrf_rewards_agreement():
    # 'b' is ranked highly by both retrievers -> should top the fused list.
    dense = _mk(["a", "b", "c"])
    lexical = _mk(["b", "d", "a"])
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_n=3)
    assert fused[0].chunk_id == "b"


def test_rrf_dedupes():
    dense = _mk(["a", "b"])
    lexical = _mk(["a", "b"])
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_n=10)
    ids = [c.chunk_id for c in fused]
    assert len(ids) == len(set(ids)) == 2
