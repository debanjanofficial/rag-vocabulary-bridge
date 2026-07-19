"""LLM-based query mapping — generative reformulation via Ollama."""

from __future__ import annotations

import re

from src.llm import chat_text
from src.terminology import load_flat_map


def _load_examples(n: int = 6) -> str:
    td = load_flat_map()
    lines = [f'  Informal: "{col}"  ->  Formal: "{formal}"'
             for col, formal in list(td.items())[:n]]
    return "\n".join(lines) if lines else "  (no examples available)"


def zero_shot(query: str) -> tuple[str, dict]:
    prompt = (
        "You are a terminology expert for REHAU RAUVISIO surface products.\n"
        "Rewrite the casual question in precise technical language as it would "
        "appear in a product manual. Return ONLY the rewritten query.\n\n"
        f"User query: {query}\nTechnical query:"
    )
    return chat_text(prompt, num_predict=120).strip(), {"method": "zero_shot"}


def few_shot(query: str) -> tuple[str, dict]:
    examples = _load_examples(6)
    prompt = (
        "You are a REHAU RAUVISIO terminology expert.\n\n"
        f"Known term mappings:\n{examples}\n\n"
        "Rewrite the casual query below in formal REHAU technical language. "
        "Return ONLY the rewritten query.\n\n"
        f"Casual query: {query}\nFormal query:"
    )
    return chat_text(prompt, num_predict=120).strip(), {"method": "few_shot", "examples": examples}


def chain_of_thought(query: str) -> tuple[str, dict]:
    examples = _load_examples(4)
    prompt = (
        "You are a REHAU RAUVISIO terminology expert.\n\n"
        f"Known term mappings:\n{examples}\n\n"
        f'A user asked: "{query}"\n\n'
        "Think step by step:\n"
        "STEP 1 - What concept is the user asking about?\n"
        "STEP 2 - Which informal terms are they using?\n"
        "STEP 3 - What is the precise REHAU/technical term?\n"
        "STEP 4 - Write the formal technical query.\n\n"
        'Output ONLY "STEP 4: <your formal query>".'
    )
    resp = chat_text(prompt, num_predict=250).strip()
    m = re.search(r"STEP\s*4\s*[:\-]?\s*(.+)", resp, re.IGNORECASE | re.DOTALL)
    if m:
        formal = m.group(1).strip().split("\n")[0].strip()
    else:
        lines = [l.strip() for l in resp.split("\n") if l.strip()]
        formal = lines[-1] if lines else query
    return formal, {"method": "cot", "full_reasoning": resp}


def multi_query(query: str, n: int = 3) -> tuple[list[str], dict]:
    prompt = (
        "You are a REHAU RAUVISIO documentation expert.\n\n"
        f"Generate {n} different formal ways to search a REHAU product manual "
        "for the answer to this question. Each should use different formal "
        "terminology. Return a numbered list, one query per line.\n\n"
        f"Original question: {query}"
    )
    resp = chat_text(prompt, num_predict=200).strip()
    variants = []
    for line in resp.split("\n"):
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip()).strip()
        if cleaned and len(cleaned) > 8:
            variants.append(cleaned)
    variants.append(query)
    variants = list(dict.fromkeys(variants))[:n + 1]
    return variants, {"method": "multi_query", "variants": variants}


def llm_map(query: str, llm_available: bool = True, method: str = "few_shot") -> dict:
    if not llm_available:
        return {
            "original_query": query, "final_query": query,
            "method_used": "passthrough (no LLM)",
            "debug": {"info": "Ollama not available"}, "strategy": "LLM-based",
        }
    try:
        if method == "zero_shot":
            result, debug = zero_shot(query)
        elif method == "cot":
            result, debug = chain_of_thought(query)
        elif method == "multi_query":
            variants, debug = multi_query(query)
            return {
                "original_query": query, "final_query": variants[0],
                "query_variants": variants, "method_used": "multi_query",
                "debug": debug, "strategy": "LLM-based",
            }
        else:
            result, debug = few_shot(query)
        return {
            "original_query": query, "final_query": result,
            "method_used": method, "debug": debug, "strategy": "LLM-based",
        }
    except Exception as e:
        return {
            "original_query": query, "final_query": query,
            "method_used": "passthrough (error)",
            "debug": {"error": str(e)}, "strategy": "LLM-based",
        }
