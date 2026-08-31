"""In-memory Okapi BM25 index over the chunked corpus.

Provides fast lexical retrieval (sub-millisecond) and Inverse Document
Frequency (IDF) calculations needed for lexical-dense agreement and
expansion drift risk estimation.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Sequence

from config import CORPUS_CHUNKS

_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Index:
    """Lightweight in-memory Okapi BM25 index for corpus chunks."""

    def __init__(
        self,
        chunk_ids: Sequence[str],
        documents: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize BM25 index with chunk IDs and corresponding texts."""
        self.k1 = k1
        self.b = b
        self.corpus_size = len(documents)
        self.chunk_ids = list(chunk_ids)

        self.doc_len: list[int] = []
        self.doc_term_freqs: list[Counter[str]] = []
        self.df: Counter[str] = Counter()

        for doc in documents:
            tokens = tokenize(doc)
            self.doc_len.append(len(tokens))
            term_counts = Counter(tokens)
            self.doc_term_freqs.append(term_counts)
            for token in term_counts:
                self.df[token] += 1

        self.avg_doc_len = (
            sum(self.doc_len) / self.corpus_size
            if self.corpus_size > 0
            else 0.0
        )

        self.idf: dict[str, float] = {}
        for term, freq in self.df.items():
            # Standard Lucene/Okapi smoothed IDF
            val = math.log(
                1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )
            self.idf[term] = max(val, 0.0)

        self.max_idf = max(self.idf.values()) if self.idf else 1.0

    def get_token_idf(self, token: str) -> float:
        """Return the precomputed IDF for a token (defaulting to max_idf)."""
        tok = token.lower()
        return self.idf.get(tok, self.max_idf)

    def query_specificity(self, query: str) -> float:
        """Compute the normalized average specificity (mean IDF) of query."""
        tokens = tokenize(query)
        if not tokens:
            return 0.0
        idfs = [self.get_token_idf(t) for t in tokens]
        mean_idf = sum(idfs) / len(idfs)
        if self.max_idf <= 0.0:
            return 0.0
        return min(max(mean_idf / self.max_idf, 0.0), 1.0)

    def search(self, query: str, top_k: int = 15) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, bm25_score) pairs for query."""
        tokens = tokenize(query)
        if not tokens or self.corpus_size == 0:
            return []

        scores = [0.0] * self.corpus_size
        for token in tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx, term_counts in enumerate(self.doc_term_freqs):
                freq = term_counts.get(token, 0)
                if freq == 0:
                    continue
                doc_l = self.doc_len[idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * (doc_l / self.avg_doc_len)
                )
                scores[idx] += idf_val * (numerator / denominator)

        # Get top-k indices with score > 0
        scored_indices = [
            (idx, score) for idx, score in enumerate(scores) if score > 0.0
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        top_results = []
        for idx, score in scored_indices[:top_k]:
            top_results.append((self.chunk_ids[idx], float(score)))
        return top_results


@lru_cache(maxsize=1)
def get_bm25_index(corpus_path: str = CORPUS_CHUNKS) -> BM25Index:
    """Load corpus chunks and build a cached singleton BM25 index."""
    if not os.path.exists(corpus_path):
        return BM25Index([], [])

    chunk_ids: list[str] = []
    documents: list[str] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            chunk_ids.append(item.get("chunk_id", ""))
            documents.append(item.get("text", ""))

    return BM25Index(chunk_ids, documents)
