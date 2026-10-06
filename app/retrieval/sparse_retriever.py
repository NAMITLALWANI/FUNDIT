"""
Sparse lexical retriever using BM25 Okapi over indexed document chunks.
"""

import re
from typing import List, Optional, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.ingestion.models import DocumentChunk

logger = get_logger(__name__)

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
}


def tokenize_text(text: str) -> List[str]:
    """Lowercase tokenization with alphanumeric filtering and stopword exclusion."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


class BM25Retriever:
    """Sparse lexical search engine for document chunks using BM25Okapi."""

    def __init__(self) -> None:
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index over provided document chunks."""
        if not chunks:
            self.chunks = []
            self.corpus_tokens = []
            self._bm25 = None
            return

        self.chunks = list(chunks)
        self.corpus_tokens = [tokenize_text(f"{c.metadata.title} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"Built BM25 index over {len(self.chunks)} chunks")

    @property
    def size(self) -> int:
        return len(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        fund_ids: Optional[Sequence[str]] = None,
        include_global: bool = True,
    ) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Execute a BM25 lexical query bounded to the candidate funds.
        Returns list of tuples: (chunk, raw_bm25_score, rank_1_based).
        """
        if self._bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        allowed = set(fund_ids) if fund_ids is not None else None
        scores = self._bm25.get_scores(query_tokens)
        scored_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: List[Tuple[DocumentChunk, float, int]] = []
        rank = 1
        for chunk_idx, score in scored_pairs:
            if score <= 0.0:  # Only include chunks with positive lexical overlap
                break
            chunk = self.chunks[chunk_idx]
            if allowed is not None:
                fid = chunk.metadata.fund_id
                if fid is None and not include_global:
                    continue
                if fid is not None and fid not in allowed:
                    continue
            results.append((chunk, float(score), rank))
            rank += 1
            if len(results) >= top_k:
                break

        return results
