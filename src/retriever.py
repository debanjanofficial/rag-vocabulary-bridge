"""ChromaDB retriever using Qwen3-Embedding on the query side."""

from __future__ import annotations

from collections import defaultdict

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME
from src.embeddings import embed_queries

_collection = None
_RRF_K = 60


def _col():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = 5) -> list[str]:
    emb = embed_queries([query]).tolist()
    results = _col().query(query_embeddings=emb, n_results=top_k, include=[])
    return results["ids"][0]


def retrieve_rrf(
    queries: list[str],
    top_k: int = 5,
    depth: int | None = None,
    rrf_k: int = _RRF_K,
) -> list[str]:
    """Fuse rankings from multiple queries via Reciprocal Rank Fusion.

    Duplicate query strings are allowed and increase that query's RRF weight
    (used to up-weight the original colloquial query).
    """
    cleaned: list[str] = []
    weight_by_key: dict[str, float] = {}
    text_by_key: dict[str, str] = {}
    for q in queries:
        key = (q or "").strip()
        if not key:
            continue
        lk = key.lower()
        if lk not in text_by_key:
            text_by_key[lk] = key
            cleaned.append(lk)
        weight_by_key[lk] = weight_by_key.get(lk, 0.0) + 1.0
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return retrieve(text_by_key[cleaned[0]], top_k=top_k)

    depth = depth or max(top_k * 3, 15)
    scores: dict[str, float] = defaultdict(float)
    texts = [text_by_key[k] for k in cleaned]
    embs = embed_queries(texts).tolist()
    for lk, emb in zip(cleaned, embs):
        w = weight_by_key[lk]
        results = _col().query(query_embeddings=[emb], n_results=depth, include=[])
        for rank, cid in enumerate(results["ids"][0]):
            scores[cid] += w / (rrf_k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def retrieve_full(query: str, top_k: int = 5) -> list[dict]:
    emb = embed_queries([query]).tolist()
    results = _col().query(
        query_embeddings=emb, n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    output = []
    for cid, doc, dist, meta in zip(
        results["ids"][0], results["documents"][0],
        results["distances"][0], results["metadatas"][0],
    ):
        output.append({
            "chunk_id": cid,
            "document": doc,
            "score": round(1 - dist, 4),
            "source": meta.get("source_pdf", ""),
            "product": meta.get("product", ""),
            "doc_type": meta.get("doc_type", ""),
        })
    return output


def retrieve_full_rrf(
    queries: list[str],
    top_k: int = 5,
    depth: int | None = None,
    rrf_k: int = _RRF_K,
) -> list[dict]:
    """Like retrieve_full, but fused across multiple query variants."""
    ids = retrieve_rrf(queries, top_k=top_k, depth=depth, rrf_k=rrf_k)
    if not ids:
        return []
    by_id: dict[str, dict] = {}
    depth = depth or max(top_k * 3, 15)
    seen_q: set[str] = set()
    for q in queries:
        key = (q or "").strip()
        if not key or key.lower() in seen_q:
            continue
        seen_q.add(key.lower())
        for doc in retrieve_full(key, top_k=depth):
            by_id.setdefault(doc["chunk_id"], doc)
    out = []
    for cid in ids:
        if cid in by_id:
            out.append(by_id[cid])
        else:
            out.append({
                "chunk_id": cid, "document": "", "score": 0.0,
                "source": "", "product": "", "doc_type": "",
            })
    return out
