"""Embedding-based query mapping — product-line semantic expansion + RRF.

Dense lookup into curated product designations only (not the full term base):
  1) Probe bank = colloquial synonyms from product_lines.json → preferred term.
  2) Embed probes once; embed the user query; rank by cosine similarity.
  3) Keep at most ONE preferred term if sim clears a strict gate; else abstain.
  4) Callers retrieve via RRF(original ×2, expanded).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from config import PRODUCT_LINES

_GATE_MIN_SIM = 0.58
_MIN_SYN_CHARS = 6

# Skip vague catch-all entries that add noise.
_SKIP_KEYS = frozenset({"general"})


def _load_product_lines() -> dict:
    with open(PRODUCT_LINES, encoding="utf-8") as f:
        return json.load(f)


def _build_probes() -> list[dict]:
    """Synonym probes only; each maps to a product preferred_term."""
    probes: list[dict] = []
    seen: set[str] = set()

    for key, entry in _load_product_lines().items():
        if key.strip().lower() in _SKIP_KEYS:
            continue
        preferred = (entry.get("preferred_term") or key).strip()
        if not preferred:
            continue
        for syn in entry.get("synonyms_colloquial") or []:
            s = (syn or "").strip()
            if len(s) < _MIN_SYN_CHARS:
                continue
            if " " not in s and "-" not in s:
                continue
            sk = s.lower()
            if sk in seen or sk == preferred.lower():
                continue
            seen.add(sk)
            probes.append({"probe": s, "preferred_term": preferred})

    return probes


@lru_cache(maxsize=1)
def _probe_index() -> tuple[tuple[dict, ...], np.ndarray]:
    from src.embeddings import embed_documents

    probes = _build_probes()
    if not probes:
        return tuple(), np.zeros((0, 1), dtype=np.float32)
    vecs = embed_documents([p["probe"] for p in probes], show_progress=False)
    return tuple(probes), vecs


def _best_product_match(query: str) -> dict | None:
    """Return best {preferred_term, probe, sim} or None if below gate / already present."""
    probes, vecs = _probe_index()
    if not probes:
        return None

    from src.embeddings import embed_queries

    q_vec = embed_queries([query])[0]
    sims = (vecs @ q_vec.reshape(-1, 1)).reshape(-1)

    q_lower = query.lower()
    best_by_pref: dict[str, dict] = {}
    for probe, sim in zip(probes, sims):
        pref = probe["preferred_term"]
        pk = pref.lower()
        if pk in q_lower:
            continue
        row = {
            "preferred_term": pref,
            "probe": probe["probe"],
            "sim": float(sim),
        }
        prev = best_by_pref.get(pk)
        if prev is None or row["sim"] > prev["sim"]:
            best_by_pref[pk] = row

    if not best_by_pref:
        return None
    best = max(best_by_pref.values(), key=lambda r: r["sim"])
    if best["sim"] < _GATE_MIN_SIM:
        return None
    return best


def embedding_map(query: str, **_ignored) -> dict:
    """Product-line semantic expansion. Extra kwargs ignored for API compat."""
    hit = _best_product_match(query)
    debug: dict = {"method": "product_line_semantic", "gate_min_sim": _GATE_MIN_SIM}

    if hit is None:
        debug["decision"] = "abstain"
        return {
            "original_query": query,
            "final_query": query,
            "appended_terms": [],
            "retrieval_queries": [query],
            "abstained": True,
            "formal_terms": [],
            "method_used": "passthrough (gate abstain)",
            "debug": debug,
            "strategy": "Embedding-based",
        }

    label = hit["preferred_term"]
    final_query = f"{query.rstrip()} ({label})"
    debug["decision"] = "expand"
    debug["kept"] = {"term": label, "probe": hit["probe"], "sim": round(hit["sim"], 4)}

    formal = [{
        "source": "synonym",
        "from": hit["probe"],
        "to": label,
        "sim": round(hit["sim"], 4),
        "adj_sim": round(hit["sim"], 4),
    }]

    return {
        "original_query": query,
        "final_query": final_query,
        "appended_terms": [label],
        "retrieval_queries": [query, query, final_query],
        "abstained": False,
        "formal_terms": formal,
        "method_used": "product_line_semantic",
        "debug": debug,
        "strategy": "Embedding-based",
    }
