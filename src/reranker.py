"""Semantic cross-encoder reranker (query–chunk relevance).

Independent of the LLM Self-Query heuristic `rerank_within_product` in
`src/retriever.py` — that path is left unchanged. This module re-scores
whatever candidate list a strategy already returned.
"""

from __future__ import annotations

from config import RERANK_MODEL, RERANK_TOP_N

_model = None


def get_reranker():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank_documents(
    query: str,
    docs: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """Re-order docs by cross-encoder relevance to ``query``.

    Adds ``rerank_score`` to each returned dict. Preserves original fields.
    """
    if not docs:
        return []

    keep = top_n if top_n is not None else RERANK_TOP_N
    keep = max(1, min(keep, len(docs)))

    pairs = [(query, (d.get("document") or "")[:2000]) for d in docs]
    scores = get_reranker().predict(pairs, show_progress_bar=False)

    ranked: list[tuple[float, dict]] = []
    for doc, score in zip(docs, scores):
        out = dict(doc)
        out["rerank_score"] = round(float(score), 4)
        ranked.append((float(score), out))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:keep]]
