"""Sentence/section-aware chunker: boundaries, overlap, and provenance propagation."""

from datetime import date

from app.ingestion.chunkers.chunker import SentenceAwareChunker
from app.ingestion.models import RawDocument


def _doc(content: str) -> RawDocument:
    return RawDocument(
        document_id="SYN-DOC-1",
        document_type="SchemeSummary",
        title="Synthetic Scheme Summary",
        source="Synthetic source",
        source_url="https://example.invalid/doc",
        fund_id="SYN-A",
        publication_date=date(2026, 9, 30),
        content=content,
        metadata={"generated": True},
    )


def test_section_aware_chunking_preserves_provenance():
    content = (
        "## Investment objective\n"
        "The scheme seeks long-term capital appreciation. It invests across market capitalisations. "
        "The objective is not guaranteed.\n\n"
        "## Costs and size\n"
        "The total expense ratio of the Direct Plan was 0.50% as of 2026-09-30. "
        "Assets under management were approximately Rs 10,000 crore."
    )
    chunks = SentenceAwareChunker(chunk_size=200, chunk_overlap=40).chunk_document(_doc(content))
    assert len(chunks) >= 2
    sections = {c.metadata.section for c in chunks}
    assert sections == {"Investment objective", "Costs and size"}
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"SYN-DOC-1-c{idx}"
        assert c.metadata.chunk_index == idx
        assert c.metadata.fund_id == "SYN-A"
        assert c.metadata.document_type == "SchemeSummary"
        assert c.metadata.source_url == "https://example.invalid/doc"
        assert c.metadata.publication_date == date(2026, 9, 30)
        assert c.metadata.extra["generated"] is True
        assert c.text.startswith(f"{c.metadata.section}: ")


def test_long_section_is_split_with_overlap():
    sentence = "Sentence number {} describes a scheme attribute in moderate detail. "
    content = "## Body\n" + "".join(sentence.format(i) for i in range(20))
    chunker = SentenceAwareChunker(chunk_size=250, chunk_overlap=80)
    chunks = chunker.chunk_document(_doc(content))
    assert len(chunks) > 2
    assert all(len(c.text) <= 250 + 80 + len("Body: ") + 80 for c in chunks)
    # overlap: the last sentence of chunk i reappears in chunk i+1
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.text.rstrip().split(". ")[-1].rstrip(".")
        assert last_sentence[:30] in b.text


def test_empty_document_yields_no_chunks():
    assert SentenceAwareChunker().chunk_document(_doc("   \n  ")) == []


def test_document_without_headings_is_single_section():
    chunks = SentenceAwareChunker(chunk_size=500).chunk_document(_doc("Plain text without any headings. Second sentence."))
    assert len(chunks) == 1
    assert chunks[0].metadata.section is None
    assert chunks[0].text == "Plain text without any headings. Second sentence."
