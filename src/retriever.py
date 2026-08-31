"""ChromaDB retriever using Qwen3-Embedding on the query side."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME
from src.embeddings import embed_queries

_collection = None
_RRF_K = 60

_BOILERPLATE = (
    "install in accordance to manufacturer's instructions",
    "install in accordance with the manufacturer's instructions",
    "install in accordance to manufacturer’s instructions",
    "install in accordance with the manufacturer’s instructions",
)


def _col():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection



def _build_where(product: str | None = None, doc_type: str | None = None) -> dict | None:
    clauses: list[dict] = []
    if product:
        clauses.append({"product": product})
    if doc_type:
        clauses.append({"doc_type": doc_type})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _is_technical_guide(source_pdf: str) -> bool:
    s = (source_pdf or "").lower().replace("\\", "/")
    return (
        "technical guide" in s
        or "instruction manual" in s
        or "/product information/" in s and "instruction" in s
    )


def _is_spec_sheet(source_pdf: str) -> bool:
    s = (source_pdf or "").lower().replace(" ", "").replace("\\", "/")
    return "specsheet" in s or "specsheets" in s


def rerank_within_product(
    docs: list[dict],
    prefer_source: str | None = None,
) -> list[dict]:
    """Intent-aware boost: guide vs spec; demote empty install pointers."""
    if not docs:
        return []
    scored: list[tuple[float, dict]] = []
    for i, d in enumerate(docs):
        score = 1.0 / (i + 1)  # preserve dense rank as base
        # Blend in embedding similarity if present
        if isinstance(d.get("score"), (int, float)):
            score += 0.5 * float(d["score"])

        src = d.get("source") or ""
        text = (d.get("document") or "").lower()

        if prefer_source == "guide":
            if _is_technical_guide(src):
                score += 2.0
            if _is_spec_sheet(src):
                score -= 1.5
        elif prefer_source == "spec":
            if _is_spec_sheet(src):
                score += 2.0
            if _is_technical_guide(src):
                score -= 0.5

        if any(b in text for b in _BOILERPLATE):
            score -= 2.0

        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


def retrieve(
    query: str,
    top_k: int = 5,
    product: str | None = None,
    doc_type: str | None = None,
) -> list[str]:
    emb = embed_queries([query]).tolist()
    kwargs: dict = {
        "query_embeddings": emb,
        "n_results": top_k,
        "include": [],
    }
    where = _build_where(product, doc_type)
    if where:
        kwargs["where"] = where
    results = _col().query(**kwargs)
    return results["ids"][0]


def retrieve_self_query(
    original_query: str,
    semantic_query: str | None,
    product: str | None = None,
    doc_type: str | None = None,
    prefer_source: str | None = None,
    top_k: int = 5,
    depth: int | None = None,
    rrf_k: int = _RRF_K,
) -> list[str]:
    """Hard product filter when set; intent-aware rerank. Else baseline."""
    del rrf_k  # unused with hard filter; kept for API compat
    depth = depth or max(top_k * 4, 20)
    search_q = (semantic_query or original_query).strip() or original_query

    if not product:
        return retrieve(original_query, top_k=top_k)

    docs = retrieve_full(
        search_q, top_k=depth, product=product, doc_type=doc_type,
    )
    # If doc_type filter emptied the pool, retry product-only.
    if not docs and doc_type:
        docs = retrieve_full(search_q, top_k=depth, product=product)

    docs = rerank_within_product(docs, prefer_source=prefer_source)
    return [d["chunk_id"] for d in docs[:top_k]]


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


def retrieve_full(
    query: str,
    top_k: int = 5,
    product: str | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    emb = embed_queries([query]).tolist()
    kwargs: dict = {
        "query_embeddings": emb,
        "n_results": top_k,
        "include": ["documents", "distances", "metadatas"],
    }
    where = _build_where(product, doc_type)
    if where:
        kwargs["where"] = where
    results = _col().query(**kwargs)
    output = []
    ids = results["ids"][0] if results["ids"] else []
    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    for cid, doc, dist, meta in zip(ids, documents, distances, metadatas):
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


def retrieve_full_self_query(
    mapped: dict,
    top_k: int = 5,
    depth: int | None = None,
) -> list[dict]:
    """Full doc payloads for Self-Query LLM retrieval (hard filter + rerank)."""
    original = mapped.get("original_query") or mapped.get("final_query") or ""
    semantic = mapped.get("semantic_query") or mapped.get("final_query") or original
    filters = mapped.get("filters") or {}
    product = filters.get("product")
    doc_type = filters.get("doc_type")
    prefer = mapped.get("prefer_source")

    depth = depth or max(top_k * 4, 20)

    if not product:
        return retrieve_full(original, top_k=top_k)

    docs = retrieve_full(semantic, top_k=depth, product=product, doc_type=doc_type)
    if not docs and doc_type:
        docs = retrieve_full(semantic, top_k=depth, product=product)
    return rerank_within_product(docs, prefer_source=prefer)[:top_k]


def retrieve_routed(
    router_result: Any,
    top_k: int = 5,
) -> list[str]:
    """Execute retrieval based on an Adaptive RouterResult object or dict."""
    if hasattr(router_result, "to_dict"):
        data = router_result.to_dict()
    else:
        data = router_result

    filters = data.get("filters") or {}
    product = filters.get("product")
    doc_type = filters.get("doc_type")
    prefer_source = data.get("prefer_source")
    orig_q = data.get("original_query", "")
    final_q = data.get("final_query", orig_q)
    queries = data.get("retrieval_queries") or [final_q]

    if product:
        return retrieve_self_query(
            original_query=orig_q,
            semantic_query=final_q,
            product=product,
            doc_type=doc_type,
            prefer_source=prefer_source,
            top_k=top_k,
        )

    if len(queries) > 1:
        return retrieve_rrf(queries, top_k=top_k)

    return retrieve(final_q or orig_q, top_k=top_k)


def retrieve_full_routed(
    router_result: Any,
    top_k: int = 5,
) -> list[dict]:
    """Execute full doc retrieval based on an Adaptive RouterResult."""
    if hasattr(router_result, "to_dict"):
        data = router_result.to_dict()
    else:
        data = router_result

    filters = data.get("filters") or {}
    product = filters.get("product")
    queries = data.get("retrieval_queries") or [data.get("final_query", "")]

    if product:
        return retrieve_full_self_query(data, top_k=top_k)

    if len(queries) > 1:
        return retrieve_full_rrf(queries, top_k=top_k)

    target_q = data.get("final_query") or data.get("original_query") or ""
    return retrieve_full(target_q, top_k=top_k)

