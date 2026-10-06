"""
Sentence-aware, section-aware document chunker preserving structural provenance and metadata.
"""

import re
from typing import List, Optional, Tuple

from app.ingestion.models import DocumentChunk, RawDocument, SourceMetadata

SECTION_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)


class SentenceAwareChunker:
    """Splits documents into coherent chunks respecting section, sentence and line boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences or coherent lines."""
        pattern = r"(?<=[.!?\n])\s+"
        parts = re.split(pattern, text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences if sentences else [text]

    @staticmethod
    def split_sections(content: str) -> List[Tuple[Optional[str], str]]:
        """Split Markdown content into (section_title, body) pairs using headings."""
        matches = list(SECTION_HEADING_RE.finditer(content))
        if not matches:
            return [(None, content)]
        sections: List[Tuple[Optional[str], str]] = []
        preamble = content[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            if body:
                sections.append((match.group(1).strip(), body))
        return sections

    def _chunk_text(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current:
                chunks.append(" ".join(current).strip())
                overlap: List[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = list(overlap)
                current_len = overlap_len
            current.append(sentence)
            current_len += sentence_len + 1

        if current:
            remaining = " ".join(current).strip()
            if not chunks or remaining != chunks[-1]:
                chunks.append(remaining)
        return chunks

    def chunk_document(self, document: RawDocument) -> List[DocumentChunk]:
        """Chunk a RawDocument into provenance-bearing DocumentChunks, one section at a time."""
        if not document.content or not document.content.strip():
            return []

        result: List[DocumentChunk] = []
        chunk_index = 0
        for section_title, body in self.split_sections(document.content):
            for chunk_text in self._chunk_text(body):
                text = f"{section_title}: {chunk_text}" if section_title else chunk_text
                metadata = SourceMetadata(
                    document_id=document.document_id,
                    document_type=document.document_type,
                    source=document.source,
                    source_url=document.source_url,
                    title=document.title,
                    fund_id=document.fund_id,
                    publication_date=document.publication_date,
                    section=section_title,
                    chunk_index=chunk_index,
                    extra=dict(document.metadata),
                )
                result.append(
                    DocumentChunk(
                        chunk_id=f"{document.document_id}-c{chunk_index}",
                        document_id=document.document_id,
                        text=text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
        return result
