"""Feature extraction pipeline for the Adaptive Vocabulary Router.

Extracts a 6-dimensional representation for each query:
  1) s_term: Terminology Gap Score (proximity to lay synonyms / preferred terms)
  2) c_prod: Product Confidence (posterior probability of top product)
  3) m_prod: Product Margin (difference between top-1 and top-2 product prob)
  4) j_agree: Lexical-Dense Agreement (Jaccard overlap between BM25 and Dense)
  5) h_dense: Retrieval Score Entropy (normalized Shannon entropy of dense top-k)
  6) r_drift: Expansion Drift Risk (penalty for expanding rare specific tokens)
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from config import PRODUCT_LINES
from src.embeddings import embed_documents, embed_queries
from src.retriever import retrieve_full
from src.router.bm25_index import get_bm25_index
from src.terminology import load_flat_map, load_terminology


@dataclass
class FeatureVector:
    """Quantitative representations for adaptive routing decisions."""

    query: str
    s_term: float
    c_prod: float
    m_prod: float
    top_product: str | None
    j_agree: float
    h_dense: float
    r_drift: float
    best_term_match: dict[str, Any] | None = None
    bm25_top_ids: list[str] | None = None
    dense_top_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert feature vector to a JSON-serializable dictionary."""
        data = asdict(self)
        data["s_term"] = round(self.s_term, 4)
        data["c_prod"] = round(self.c_prod, 4)
        data["m_prod"] = round(self.m_prod, 4)
        data["j_agree"] = round(self.j_agree, 4)
        data["h_dense"] = round(self.h_dense, 4)
        data["r_drift"] = round(self.r_drift, 4)
        return data


def _load_product_lines_data() -> dict[str, Any]:
    """Load product lines dictionary from JSON config."""
    if not os.path.exists(PRODUCT_LINES):
        return {}
    with open(PRODUCT_LINES, encoding="utf-8") as file_obj:
        return json.load(file_obj)


@lru_cache(maxsize=1)
def _product_probes_index() -> tuple[list[tuple[str, str]], np.ndarray]:
    """Build product alias probes and embeddings for product classification."""
    data = _load_product_lines_data()
    probes: list[tuple[str, str]] = []  # (alias_text, canonical_product)

    for key, entry in data.items():
        if key.strip().lower() == "general":
            continue
        preferred = (entry.get("preferred_term") or key).strip()
        synonyms = entry.get("synonyms_colloquial") or []
        for syn in synonyms:
            s_clean = (syn or "").strip()
            if len(s_clean) >= 3:
                probes.append((s_clean, preferred))
        probes.append((preferred, preferred))

    if not probes:
        return [], np.zeros((0, 1), dtype=np.float32)

    texts = [p[0] for p in probes]
    vectors = embed_documents(texts, show_progress=False)
    return probes, vectors


@lru_cache(maxsize=1)
def _terminology_probes_index() -> tuple[list[dict[str, str]], np.ndarray]:
    """Precompute embeddings for flat_map and terminology concept phrases."""
    flat_map = load_flat_map()
    probes: list[dict[str, str]] = []

    for phrase, preferred in flat_map.items():
        p_clean = phrase.strip()
        if len(p_clean) >= 3:
            probes.append({
                "phrase": p_clean,
                "preferred_term": preferred.strip(),
            })

    if not probes:
        return [], np.zeros((0, 1), dtype=np.float32)

    texts = [p["phrase"] for p in probes]
    vectors = embed_documents(texts, show_progress=False)
    return probes, vectors


def compute_terminology_gap(
    query: str,
    query_vec: np.ndarray,
) -> tuple[float, dict[str, Any] | None]:
    """Calculate the Terminology Gap score S_term in [0, 1].

    Checks exact phrase substrings first, followed by dense probe cosine
    similarity against the curated terminology base.
    """
    flat_map = load_flat_map()
    query_lower = query.lower()

    # Exact dictionary substring match check (score = 1.0)
    # Exclude broad catalog brand names and generic container nouns
    excluded_terms = {"rauvisio", "product", "material", "board"}
    for phrase, preferred in flat_map.items():
        p_low = phrase.lower().strip()
        if p_low in excluded_terms:
            continue
        pattern = r"\b" + re.escape(p_low) + r"\b"
        if len(p_low) >= 4 and re.search(pattern, query_lower):
            return 1.0, {
                "phrase": phrase,
                "preferred_term": preferred,
                "match_type": "exact",
                "sim": 1.0,
            }

    probes, vectors = _terminology_probes_index()
    if not probes or vectors.shape[0] == 0:
        return 0.0, None

    sims = (vectors @ query_vec.reshape(-1, 1)).reshape(-1)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim > 0.35:
        match_info = {
            "phrase": probes[best_idx]["phrase"],
            "preferred_term": probes[best_idx]["preferred_term"],
            "match_type": "dense",
            "sim": best_sim,
        }
        return min(max(best_sim, 0.0), 1.0), match_info

    return 0.0, None


def compute_product_confidence(
    query: str,
    query_vec: np.ndarray,
    temperature: float = 0.08,
) -> tuple[float, float, str | None]:
    """Compute product classification confidence (C_prod) and margin (M_prod).

    Returns:
        (c_prod, m_prod, top_product_name)
    """
    data = _load_product_lines_data()
    all_products = [
        (entry.get("preferred_term") or key).strip()
        for key, entry in data.items()
        if key.strip().lower() != "general"
    ]
    if not all_products:
        return 0.0, 0.0, None

    query_lower = query.lower()
    # Direct mention bonus
    for prod in all_products:
        if prod.lower() in query_lower:
            return 0.99, 0.90, prod

    probes, vectors = _product_probes_index()
    if not probes or vectors.shape[0] == 0:
        return 0.0, 0.0, None

    sims = (vectors @ query_vec.reshape(-1, 1)).reshape(-1)

    # Aggregate max similarity per product
    product_scores: dict[str, float] = {p: 0.0 for p in all_products}
    for (alias, prod), sim in zip(probes, sims):
        if sim > product_scores[prod]:
            product_scores[prod] = float(sim)

    score_vals = np.array([product_scores[p] for p in all_products])
    # Softmax with temperature
    exp_scores = np.exp((score_vals - np.max(score_vals)) / temperature)
    probs = exp_scores / np.sum(exp_scores)

    ranked_indices = np.argsort(probs)[::-1]
    top_idx = ranked_indices[0]
    second_idx = ranked_indices[1] if len(ranked_indices) > 1 else None

    c_prod = float(probs[top_idx])
    m_prod = (
        float(probs[top_idx] - probs[second_idx])
        if second_idx is not None
        else c_prod
    )
    top_product = all_products[top_idx]

    return c_prod, m_prod, top_product


def compute_lexical_dense_agreement(
    query: str,
    top_k: int = 15,
) -> tuple[float, list[str], list[str]]:
    """Compute Jaccard agreement between in-memory BM25 and dense retrieval."""
    bm25_idx = get_bm25_index()
    bm25_hits = bm25_idx.search(query, top_k=top_k)
    bm25_ids = [cid for cid, _ in bm25_hits]

    dense_hits = retrieve_full(query, top_k=top_k)
    dense_ids = [d["chunk_id"] for d in dense_hits]

    if not bm25_ids or not dense_ids:
        return 0.0, bm25_ids, dense_ids

    set_bm25 = set(bm25_ids)
    set_dense = set(dense_ids)

    intersection = len(set_bm25 & set_dense)
    union = len(set_bm25 | set_dense)

    jaccard = intersection / union if union > 0 else 0.0
    return float(jaccard), bm25_ids, dense_ids


def compute_retrieval_entropy(
    dense_hits: list[dict[str, Any]],
    top_k: int = 20,
    temperature: float = 0.1,
) -> float:
    """Compute normalized Shannon entropy H_dense over top-k dense scores."""
    if not dense_hits:
        return 1.0

    scores = [
        float(d.get("score", 0.0)) for d in dense_hits[:top_k]
    ]
    if len(scores) <= 1:
        return 0.0

    arr = np.array(scores, dtype=np.float64)
    # Softmax over scores
    exp_arr = np.exp((arr - np.max(arr)) / temperature)
    probs = exp_arr / np.sum(exp_arr)

    # Avoid log(0)
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = math.log(len(scores))

    normalized_entropy = (
        entropy / max_entropy if max_entropy > 0.0 else 0.0
    )
    return float(min(max(normalized_entropy, 0.0), 1.0))


def compute_expansion_drift_risk(query: str, s_term: float) -> float:
    """Compute expansion drift risk R_drift in [0, 1].

    Penalizes query expansion when the query contains highly specific rare
    terms but has low similarity to the terminology base.
    """
    bm25_idx = get_bm25_index()
    specificity = bm25_idx.query_specificity(query)
    risk = specificity * (1.0 - s_term)
    return float(min(max(risk, 0.0), 1.0))


def extract_features(query: str) -> FeatureVector:
    """Extract complete quantitative FeatureVector for a given query."""
    q_clean = (query or "").strip()
    if not q_clean:
        return FeatureVector(
            query="",
            s_term=0.0,
            c_prod=0.0,
            m_prod=0.0,
            top_product=None,
            j_agree=0.0,
            h_dense=1.0,
            r_drift=0.0,
            best_term_match=None,
            bm25_top_ids=[],
            dense_top_ids=[],
        )

    # Embed query once
    query_vec = embed_queries([q_clean])[0]

    s_term, term_match = compute_terminology_gap(q_clean, query_vec)
    c_prod, m_prod, top_product = compute_product_confidence(
        q_clean, query_vec
    )
    j_agree, bm25_ids, dense_ids = compute_lexical_dense_agreement(
        q_clean, top_k=15
    )

    # Dense hits for entropy calculation
    dense_hits = retrieve_full(q_clean, top_k=20)
    h_dense = compute_retrieval_entropy(dense_hits, top_k=20)
    r_drift = compute_expansion_drift_risk(q_clean, s_term)

    return FeatureVector(
        query=q_clean,
        s_term=s_term,
        c_prod=c_prod,
        m_prod=m_prod,
        top_product=top_product,
        j_agree=j_agree,
        h_dense=h_dense,
        r_drift=r_drift,
        best_term_match=term_match,
        bm25_top_ids=bm25_ids,
        dense_top_ids=dense_ids,
    )
