from src.rag.chunking import PageText, chunk_pages, _split_with_overlap


def test_section_detection_and_pages():
    pages = [
        PageText(1, "1 Purpose\nThis is the purpose section.\n2 Scope\nScope text here."),
        PageText(2, "4.2 EDD Measures\nSenior management approval is required."),
    ]
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=100)
    labels = {c["section"] for c in chunks}
    assert "§1" in labels
    assert "§2" in labels
    assert "§4.2" in labels
    # §4.2 content originates on page 2
    edd = [c for c in chunks if c["section"] == "§4.2"][0]
    assert edd["page"] == 2


def test_overlap_splits_long_text():
    text = "Sentence. " * 400  # ~4000 chars
    pieces = _split_with_overlap(text, size=900, overlap=150)
    assert len(pieces) > 1
    assert all(len(p) <= 950 for p in pieces)


def test_short_text_single_chunk():
    assert _split_with_overlap("short text", 900, 150) == ["short text"]
