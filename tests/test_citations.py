from src.rag.schemas import Chunk, ScoredChunk
from src.rag.citations import build_citations, used_markers, filter_to_used


def _sc(section, page, doc="AML Policy"):
    ch = Chunk("id1", "Some passage text about due diligence.", "AML.pdf", doc, section, page)
    return ScoredChunk(chunk=ch, score=0.9)


def test_citation_format():
    ch = Chunk("x", "t", "AML-Policy.pdf", "AML Policy", "§4.2", 3)
    assert ch.citation() == "AML Policy §4.2, page 3"


def test_build_and_filter_citations():
    scored = [_sc("§4.1", 2), _sc("§4.2", 2)]
    cites = build_citations(scored)
    assert cites[0].marker == "[1]" and cites[1].marker == "[2]"
    # answer only uses [2] -> filter keeps just that one
    kept = filter_to_used("The rule is X [2].", cites)
    assert len(kept) == 1 and kept[0].marker == "[2]"


def test_used_markers():
    assert used_markers("foo [1] bar [3] baz [1]") == {1, 3}
