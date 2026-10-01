"""Semantic cross-encoder reranker (query-chunk relevance).

Supports Path-Injected Neural Reranking by incorporating multi-hop
graph traversal provenance from the Terminology Knowledge Graph (TKG)
directly into the cross-attention layers of the CrossEncoder model.
"""

from __future__ import annotations

from typing import Any, Sequence

from config import RERANK_MODEL, RERANK_TOP_N

_model = None


def get_reranker():
    """Lazily load cross-encoder model checkpoint."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank_documents(
    query: str,
    docs: list[dict],
    top_n: int | None = None,
    path_context: Any = None,
    provenance_chunk_ids: Sequence[str] | set[str] | None = None,
    provenance_boost: float = 0.05,
) -> list[dict]:
    """Re-order docs by cross-encoder relevance with optional path injection.

    Args:
        query: Inbound user question string.
        docs: Candidate document passages from lexical/dense retrieval.
        top_n: Number of top-ranked candidates to retain.
        path_context: SubgraphResult instance, string, or list of path
            strings representing multi-hop TKG traversal provenance.
        provenance_chunk_ids: Chunk IDs directly grounded in the TKG.
        provenance_boost: Score bonus for candidates grounded in graph.

    Returns:
        Sorted list of candidate dicts with calibrated ``rerank_score``.
    """
    if not docs:
        return []

    keep = top_n if top_n is not None else RERANK_TOP_N
    keep = max(1, min(keep, len(docs)))

    # Extract standardized string representation of path context
    clean_path = ""
    prov_set: set[str] = set()

    if path_context is not None:
        if hasattr(path_context, "get_path_context"):
            clean_path = path_context.get_path_context()
            if provenance_chunk_ids is None and hasattr(
                path_context, "provenance_chunk_ids"
            ):
                prov_set = set(path_context.provenance_chunk_ids)
        elif isinstance(path_context, (list, tuple)):
            filtered = [str(p).strip() for p in path_context if str(p).strip()]
            if filtered:
                clean_path = f"[Path: {' | '.join(filtered[:3])}]"
        elif isinstance(path_context, str):
            clean_path = path_context.strip()
            if clean_path and not clean_path.startswith("[Path:"):
                clean_path = f"[Path: {clean_path}]"

    if provenance_chunk_ids is not None:
        prov_set = set(provenance_chunk_ids)

    # Inject path context into query sequence for cross-attention
    if clean_path:
        rerank_query = f"{query} {clean_path}".strip()
    else:
        rerank_query = query

    pairs = [(rerank_query, (d.get("document") or "")[:2000]) for d in docs]
    scores = get_reranker().predict(pairs, show_progress_bar=False)

    ranked: list[tuple[float, dict]] = []
    for doc, score in zip(docs, scores):
        out = dict(doc)
        base_score = float(score)

        cid = doc.get("chunk_id") or doc.get("id") or ""
        is_grounded = bool(cid and cid in prov_set)
        boost = provenance_boost if is_grounded else 0.0

        final_score = base_score + boost
        out["rerank_score"] = round(final_score, 4)
        out["base_cross_encoder_score"] = round(base_score, 4)
        out["graph_provenance"] = is_grounded
        if clean_path:
            out["path_context"] = clean_path

        ranked.append((final_score, out))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:keep]]
