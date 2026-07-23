"""Grounded answer generation from reranked chunks (plain-language, small model)."""

from __future__ import annotations

from config import (
    ANSWER_MODEL,
    GENERATE_MAX_CHARS,
    GENERATE_MAX_CHARS_DETAILED,
    GENERATE_NUM_PREDICT,
    GENERATE_NUM_PREDICT_DETAILED,
    GENERATE_TOP_N,
    GENERATE_TOP_N_DETAILED,
)
from src.llm import chat_text


def _format_contexts(docs: list[dict], max_chars_per_doc: int) -> str:
    blocks: list[str] = []
    for i, d in enumerate(docs, 1):
        cid = d.get("chunk_id", f"doc_{i}")
        product = d.get("product") or ""
        text = (d.get("document") or "").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "…"
        header = f"[{i}] {cid}"
        if product:
            header += f" ({product})"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _prompt_for_mode(query: str, context: str, detail: bool) -> str:
    if detail:
        return (
            "You are a REHAU RAUVISIO helper for shopfloor staff.\n"
            "Answer using ONLY the passages. Everyday language. Keep numbers/standards exact.\n"
            "Match answer LENGTH to the question:\n"
            "- Simple facts (ratings, sizes, yes/no, one value) → short and to the point. "
            "Do NOT pad or add filler.\n"
            "- How-to / multi-step / safety procedures → give complete steps, numbers, "
            "warnings, and product names from the passages.\n"
            "Never invent facts. If passages are not enough, say what is missing.\n"
            "End with: Sources: [n], ...\n\n"
            f"Question: {query}\n\n"
            f"Passages:\n{context}\n\n"
            "Answer:"
        )
    return (
        "Answer using ONLY the passages. Everyday language. Keep numbers/standards exact.\n"
        "Be concise (a short paragraph or a few bullets). Do not invent facts.\n"
        "If passages are not enough, say you do not know.\n"
        "End with: Sources: [n], ...\n\n"
        f"Question: {query}\n\n"
        f"Passages:\n{context}\n\n"
        "Answer:"
    )


def generate_answer(
    query: str,
    docs: list[dict],
    llm_available: bool = True,
    top_n: int | None = None,
    model: str | None = None,
    detail: bool = False,
) -> dict:
    """Answer ``query`` with ``qwen3:0.6b`` (short or detailed).

    ``detail=True`` gives more passage context and room for long replies when
    needed; the model is instructed to stay brief for simple fact questions.
    """
    if detail:
        default_n = GENERATE_TOP_N_DETAILED
        max_chars = GENERATE_MAX_CHARS_DETAILED
        num_predict = GENERATE_NUM_PREDICT_DETAILED
    else:
        default_n = GENERATE_TOP_N
        max_chars = GENERATE_MAX_CHARS
        num_predict = GENERATE_NUM_PREDICT

    keep = top_n if top_n is not None else default_n
    docs = docs[: max(1, keep)] if docs else []
    used_ids = [d.get("chunk_id", "") for d in docs if d.get("chunk_id")]
    answer_model = model or ANSWER_MODEL

    if not docs:
        return {
            "answer": "No documents were retrieved, so I cannot answer.",
            "abstained": True,
            "citations": [],
            "used_chunk_ids": [],
            "error": None,
            "model": answer_model,
            "detail": detail,
        }

    if not llm_available:
        return {
            "answer": (
                f"Answer generation needs Ollama model `{answer_model}`. "
                "Retrieved and reranked passages are shown above."
            ),
            "abstained": True,
            "citations": used_ids[:5],
            "used_chunk_ids": used_ids,
            "error": "llm_unavailable",
            "model": answer_model,
            "detail": detail,
        }

    context = _format_contexts(docs, max_chars)
    prompt = _prompt_for_mode(query, context, detail)

    try:
        answer = chat_text(
            prompt,
            model=answer_model,
            temperature=0.2,
            num_predict=num_predict,
        ).strip()
    except Exception as e:
        return {
            "answer": f"Generation failed: {e}",
            "abstained": True,
            "citations": [],
            "used_chunk_ids": used_ids,
            "error": str(e),
            "model": answer_model,
            "detail": detail,
        }

    low = answer.lower()
    abstained = any(
        p in low
        for p in (
            "not enough",
            "insufficient",
            "cannot answer",
            "can't answer",
            "do not have enough",
            "don't have enough",
            "do not know",
            "don't know",
            "no information",
            "missing information",
        )
    )

    return {
        "answer": answer,
        "abstained": abstained,
        "citations": used_ids,
        "used_chunk_ids": used_ids,
        "error": None,
        "model": answer_model,
        "detail": detail,
    }
