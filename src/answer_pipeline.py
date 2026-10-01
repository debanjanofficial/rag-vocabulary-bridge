"""Selected-strategy retrieve → semantic rerank → grounded answer.

Leaves LLM Self-Query's internal `rerank_within_product` untouched; this
pipeline applies an additional cross-encoder pass for every strategy.
"""

from __future__ import annotations

from config import (
    GENERATE_TOP_N,
    GENERATE_TOP_N_DETAILED,
    RETRIEVE_CANDIDATES,
    RERANK_TOP_N,
)
from src.generator import generate_answer
from src.reranker import rerank_documents
from src.retriever import (
    retrieve_full,
    retrieve_full_rrf,
    retrieve_full_self_query,
)


def retrieve_candidates(
    strategy: str,
    query: str,
    mapped: dict,
    candidate_k: int | None = None,
) -> list[dict]:
    """Fetch a wide candidate pool for the selected strategy only."""
    k = candidate_k or RETRIEVE_CANDIDATES
    k = max(k, RERANK_TOP_N)

    if strategy in ("Adaptive (Auto-Route)", "Adaptive"):
        from src.retriever import retrieve_full_routed
        return retrieve_full_routed(mapped, top_k=k)
    if strategy in ("Hybrid (BM25 + Dense RRF)", "Hybrid"):
        from src.retriever import retrieve_full_hybrid
        return retrieve_full_hybrid(
            query=mapped.get("final_query") or query,
            lexical_query=mapped.get("lexical_query"),
            alpha=float(mapped.get("alpha", 0.5)),
            top_k=k,
        )
    if strategy == "LLM-based":
        # Internal heuristic guide/spec rerank stays inside this call.
        return retrieve_full_self_query(mapped, top_k=k)
    if strategy in ("NLP-based", "Embedding-based"):
        return retrieve_full_rrf(
            mapped.get("retrieval_queries") or [query],
            top_k=k,
        )
    q = mapped.get("final_query") or query
    return retrieve_full(q, top_k=k)



def run_answer_pipeline(
    strategy: str,
    query: str,
    mapped: dict,
    llm_available: bool = True,
    candidate_k: int | None = None,
    rerank_top_n: int | None = None,
    generate: bool = True,
    generate_top_n: int | None = None,
    detail: bool = False,
    verify_grounding: bool = True,
) -> dict:
    """Post-mapping path for one user-selected strategy.

    When ``generate`` is False, skips Ollama answer generation.
    ``detail=True`` uses more context; answer length matches the question.
    ``verify_grounding=True`` executes sentence-level NLI checking.
    """
    top_n = rerank_top_n or RERANK_TOP_N
    if generate_top_n is not None:
        gen_n = generate_top_n
    else:
        gen_n = GENERATE_TOP_N_DETAILED if detail else GENERATE_TOP_N

    candidates = retrieve_candidates(
        strategy, query, mapped, candidate_k=candidate_k
    )
    path_ctx = (
        mapped.get("path_context")
        or mapped.get("subgraph_result")
        if isinstance(mapped, dict)
        else None
    )
    reranked = rerank_documents(
        query, candidates, top_n=top_n, path_context=path_ctx
    )

    if generate:
        generation = generate_answer(
            query,
            reranked,
            llm_available=llm_available,
            top_n=gen_n,
            detail=detail,
            verify_grounding=verify_grounding,
        )
    else:
        top_cids = [d.get("chunk_id", "") for d in reranked[:gen_n]]
        generation = {
            "answer": "Answer generation skipped (toggle off).",
            "abstained": True,
            "citations": top_cids,
            "used_chunk_ids": top_cids,
            "error": "skipped",
            "model": None,
            "detail": detail,
        }

    return {
        "candidates": candidates,
        "reranked": reranked,
        "generation": generation,
        "generate_top_n": gen_n,
        "detail": detail,
    }
