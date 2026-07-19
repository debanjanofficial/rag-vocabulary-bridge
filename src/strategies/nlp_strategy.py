"""NLP-based query mapping — in-place parenthetical formal terms.

Pipeline:
  1) Collect candidates from multi-word flat_map hits and phrase-grounded
     concept matches (preferred terms only — no characteristic prose).
  2) Embedding-gate candidates against the query (abstain if none pass).
  3) Insert gated terms as ``matched phrase (Formal Term)`` in-place.
  4) Retrieve via RRF over the original query (weighted) plus the expanded
     query so weak expansions cannot erase baseline hits.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from src.terminology import load_flat_map, load_terminology

_MAX_FLAT_HITS = 5
_MAX_CONCEPTS = 4
_MAX_APPENDED = 2
_MIN_CONCEPT_SCORE = 3.5
_MIN_PHRASE_CHARS = 6

# Embedding gate (cosine on L2-normalized vectors).
_GATE_MIN_SIM = 0.45
_GATE_REL_MARGIN = 0.06
_FLAT_SIM_BOOST = 0.04
_PHRASE_SIM_BOOST = 0.03
_MAX_TERM_TOKENS = 4  # drop long procedure-style preferred terms (not flat_map)

_WEAK_PROCESS = frozenset({
    "press", "pressing", "cut", "cutting", "edge", "edges",
    "install", "installation", "installing", "work", "working",
})

_STOPWORDS = frozenset({
    "i", "my", "me", "can", "do", "just", "some", "a", "an", "the", "to",
    "of", "in", "on", "it", "is", "be", "are", "was", "for", "with", "that",
    "this", "these", "those", "and", "or", "not", "if", "how", "what",
    "where", "when", "why", "which", "who", "use", "used", "will", "would",
    "should", "could", "get", "got", "let", "did", "its", "at", "by", "as",
    "from", "into", "about", "than", "then", "them", "they", "we", "you",
    "your", "our", "out", "up", "down", "over", "under", "also", "any",
    "all", "make", "sure", "look", "good", "best", "way", "kind", "kinds",
    "available", "stuff", "thing", "things", "does", "dont", "don't",
    "properly", "using", "like", "need", "needs", "stop", "keep", "ready",
    "work", "working",
})

_PRODUCT_MARKERS = (
    "brilliant", "crystal", "noir", "shade", "ferro", "ingrain", "terra",
    "floating", "laseredge",
)

_phrase_patterns: list[tuple[re.Pattern[str], str, str]] | None = None


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Match multi-word phrase with simple plurals (no word skipping)."""
    tokens = phrase.lower().replace("-", " ").split()
    parts: list[str] = []
    for i, token in enumerate(tokens):
        stem = re.escape(token.rstrip("s"))
        parts.append(stem + r"(?:s)?")
        if i < len(tokens) - 1:
            parts.append(r"[\s\-]+")
    return re.compile(r"(?<!\w)" + "".join(parts) + r"(?!\w)", re.IGNORECASE)


def _is_safe_phrase(phrase: str) -> bool:
    p = phrase.strip().lower()
    if len(p) < _MIN_PHRASE_CHARS:
        return False
    if " " not in p and "-" not in p:
        return False
    if p in _STOPWORDS or p in _WEAK_PROCESS:
        return False
    return True


def _detect_product_hint(text: str) -> str | None:
    tl = text.lower()
    for p in _PRODUCT_MARKERS:
        if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", tl):
            return p
    return None


def _short_label(preferred_term: str) -> str:
    """Keep short product names intact; trim long procedure-style designations."""
    parts = preferred_term.strip().split()
    if len(parts) <= 3:
        return preferred_term.strip()
    if parts[0].lower() in ("rauvisio", "rehau"):
        return f"{parts[0]} {parts[1]}"
    if len(preferred_term) <= 40:
        return preferred_term.strip()
    return " ".join(parts[:3])


# ── Layer 1: flat_map hits ────────────────────────────────────────────────────

def _build_phrase_patterns(term_dict: dict[str, str]) -> list[tuple[re.Pattern[str], str, str]]:
    entries: list[tuple[re.Pattern[str], str, str]] = []
    for colloquial, formal in sorted(term_dict.items(), key=lambda x: len(x[0]), reverse=True):
        if not _is_safe_phrase(colloquial):
            continue
        key = colloquial.replace("-", " ")
        entries.append((_phrase_pattern(key), colloquial, formal))
    return entries


def _get_phrase_patterns() -> list[tuple[re.Pattern[str], str, str]]:
    global _phrase_patterns
    if _phrase_patterns is None:
        _phrase_patterns = _build_phrase_patterns(load_flat_map())
    return _phrase_patterns


def detect_flat_map_hits(query: str) -> list[dict]:
    """Find multi-word flat_map phrases; return formal terms for expansion."""
    hits: list[dict] = []
    seen: set[str] = set()
    for pattern, colloquial, formal in _get_phrase_patterns():
        m = pattern.search(query)
        if not m:
            continue
        key = formal.lower().strip()
        if key in seen or key in query.lower():
            continue
        hits.append({
            "from": colloquial,
            "to": formal,
            "matched": m.group(0),
            "layer": "flat_map",
        })
        seen.add(key)
        if len(hits) >= _MAX_FLAT_HITS:
            break
    return hits


# ── Layer 2: Phrase-grounded concept matching ─────────────────────────────────

@lru_cache(maxsize=1)
def _concept_index() -> list[dict]:
    indexed: list[dict] = []
    for c in load_terminology()["concepts"]:
        preferred = (c.get("preferred_term") or "").strip()
        if not preferred or ":" in preferred:
            continue
        # Skip noisy procedure/spec designations (digits, too many tokens)
        if any(ch.isdigit() for ch in preferred):
            continue
        if len(preferred.split()) > _MAX_TERM_TOKENS:
            continue
        syns = [s.strip() for s in (c.get("synonyms_colloquial") or []) if s.strip()]
        acronyms = [a.strip() for a in (c.get("acronyms") or []) if a.strip()]
        synonym_phrases = [
            s.lower().replace("-", " ").strip()
            for s in syns + acronyms
            if _is_safe_phrase(s)
        ]
        if not synonym_phrases:
            continue
        indexed.append({
            "concept_id": c.get("concept_id", ""),
            "preferred_term": preferred,
            "synonym_phrases": synonym_phrases,
            "product": _detect_product_hint(preferred),
        })
    return indexed


def match_concepts(
    query: str,
    top_k: int = _MAX_CONCEPTS,
    product_hint: str | None = None,
) -> list[dict]:
    """Match concepts only via multi-word synonym phrases (high precision)."""
    product_hint = product_hint or _detect_product_hint(query)
    scored: list[dict] = []

    for c in _concept_index():
        matched_phrase = None
        for phrase in c["synonym_phrases"]:
            m = _phrase_pattern(phrase).search(query)
            if m:
                matched_phrase = m.group(0)
                break
        if not matched_phrase:
            continue

        score = 5.5
        reasons = [f"synonym: {matched_phrase}"]
        if product_hint and c["product"]:
            if c["product"] == product_hint:
                score *= 1.35
                reasons.append(f"product match: {product_hint}")
            else:
                score *= 0.45

        if score < _MIN_CONCEPT_SCORE:
            continue

        scored.append({
            "concept_id": c["concept_id"],
            "preferred_term": c["preferred_term"],
            "matched_phrase": matched_phrase,
            "synonym_phrases": c["synonym_phrases"],
            "score": round(score, 2),
            "reasons": reasons,
            "already_in_query": c["preferred_term"].lower() in query.lower(),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    seen: set[str] = set()
    unique: list[dict] = []
    for hit in scored:
        key = hit["preferred_term"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
        if len(unique) >= top_k:
            break
    return unique


# ── Layer 3: Candidates + embedding gate + in-place expansion ─────────────────

def _concept_has_phrase_hit(concept: dict) -> bool:
    return any(r.startswith("synonym:") for r in concept.get("reasons") or [])


def collect_candidates(
    query: str,
    flat_hits: list[dict],
    concepts: list[dict],
) -> list[dict]:
    """Preferred-term candidates only (no characteristic / definition prose)."""
    q_lower = query.lower()
    out: list[dict] = []
    seen: set[str] = set()

    def _add(term: str, source: str, from_text: str, priority: float,
             matched: str | None = None) -> None:
        t = term.strip()
        if not t:
            return
        if source != "flat_map" and len(t.split()) > _MAX_TERM_TOKENS:
            return
        key = t.lower()
        if key in seen or key in q_lower:
            return
        seen.add(key)
        out.append({
            "term": t,
            "source": source,
            "from": from_text,
            "matched": matched or from_text,
            "priority": priority,
        })

    for h in flat_hits:
        _add(
            h["to"], "flat_map", h["from"], priority=2.0,
            matched=h.get("matched") or h["from"],
        )

    flat_spans = {
        (h.get("matched") or h["from"]).lower().strip()
        for h in flat_hits
    }

    for c in concepts:
        if not _concept_has_phrase_hit(c):
            continue
        pt = c["preferred_term"]
        phrase = c.get("matched_phrase") or "(matched)"
        # flat_map already owns this span — don't double-map to a noisier concept
        if phrase.lower().strip() in flat_spans:
            continue
        pri = 1.5 + min(0.5, float(c.get("score") or 0.0) / 100.0)
        _add(pt, "concept", phrase, priority=pri, matched=phrase)

    return out


def _cosine_rows(query_vec: np.ndarray, term_vecs: np.ndarray) -> np.ndarray:
    q = query_vec.reshape(1, -1)
    return (term_vecs @ q.T).reshape(-1)


def gate_candidates(query: str, candidates: list[dict]) -> tuple[list[dict], dict]:
    """Keep only embedding-aligned terms; abstain if none clear the gate."""
    debug: dict = {"method": "embedding_gate", "candidates": []}
    if not candidates:
        debug["decision"] = "abstain_empty"
        return [], debug

    from src.embeddings import embed_documents, embed_queries

    q_vec = embed_queries([query])[0]
    terms = [c["term"] for c in candidates]
    t_vecs = embed_documents(terms)
    sims = _cosine_rows(q_vec, t_vecs)

    scored: list[dict] = []
    for cand, sim in zip(candidates, sims):
        adj = float(sim)
        if cand["source"] == "flat_map":
            adj += _FLAT_SIM_BOOST
        else:
            adj += _PHRASE_SIM_BOOST
        row = {
            **cand,
            "sim": round(float(sim), 4),
            "adj_sim": round(adj, 4),
            "min_needed": _GATE_MIN_SIM,
        }
        scored.append(row)
        debug["candidates"].append(row)

    scored.sort(key=lambda x: (x["adj_sim"], x["priority"]), reverse=True)
    best = scored[0]["adj_sim"]
    kept: list[dict] = []
    for row in scored:
        if row["adj_sim"] < _GATE_MIN_SIM:
            continue
        if row["adj_sim"] < best - _GATE_REL_MARGIN:
            continue
        kept.append(row)
        if len(kept) >= _MAX_APPENDED:
            break

    if not kept:
        debug["decision"] = "abstain_gate"
        debug["best_adj_sim"] = best
        return [], debug

    debug["decision"] = "expand"
    debug["kept"] = [k["term"] for k in kept]
    return kept, debug


def _display_term(term: str) -> str:
    """Shorten long controlled-vocabulary strings for parenthetical display."""
    if term.lower().startswith(("rauvisio ", "rehau ")):
        return _short_label(term)
    if len(term.split()) > _MAX_TERM_TOKENS:
        return _short_label(term)
    return term


def expand_with_parentheses(query: str, gated: list[dict]) -> str:
    """Insert ``(Formal Term)`` immediately after each matched colloquial span."""
    if not gated:
        return query

    result = query
    ordered = sorted(
        gated,
        key=lambda g: len(g.get("matched") or g.get("from") or ""),
        reverse=True,
    )
    for g in ordered:
        term = _display_term(g.get("term") or "")
        if not term:
            continue
        if re.search(rf"\(\s*{re.escape(term)}\s*\)", result, re.IGNORECASE):
            continue
        if term.lower() in result.lower():
            continue

        anchor = (g.get("matched") or g.get("from") or "").strip()
        inserted = False
        if anchor and anchor != "(matched)":
            pattern = _phrase_pattern(anchor)
            already = re.compile(
                pattern.pattern + r"\s*\([^)]+\)",
                re.IGNORECASE,
            )
            if not already.search(result):
                def _repl(m: re.Match[str], t: str = term) -> str:
                    return f"{m.group(0)} ({t})"

                new_result, n = pattern.subn(_repl, result, count=1)
                if n:
                    result = new_result
                    inserted = True

        if not inserted:
            result = f"{result.rstrip()} ({term})"

    return result.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def nlp_map(query: str) -> dict:
    flat_hits = detect_flat_map_hits(query)
    product_hint = _detect_product_hint(query)
    concepts = match_concepts(query, product_hint=product_hint, top_k=_MAX_CONCEPTS)

    candidates = collect_candidates(query, flat_hits, concepts)
    gated, gate_debug = gate_candidates(query, candidates)
    # Display/insert uses shortened labels; gating used full preferred terms
    appended = [_display_term(g["term"]) for g in gated]
    final_query = expand_with_parentheses(query, gated)

    formal_terms = [
        {"source": g["source"], "from": g["from"], "to": _display_term(g["term"]),
         "sim": g.get("sim"), "adj_sim": g.get("adj_sim")}
        for g in gated
    ]

    # Dual-query RRF: original (weighted x2) + parenthetical expansion
    retrieval_queries = [query, query]
    if appended and final_query.strip() != query.strip():
        retrieval_queries.append(final_query)

    return {
        "original_query": query,
        "display_query": query,
        "final_query": final_query.strip(),
        "appended_terms": appended,
        "retrieval_queries": retrieval_queries,
        "abstained": not appended,
        "gate": gate_debug,
        "substitutions": flat_hits,
        "synonym_hits": flat_hits,
        "formal_terms": formal_terms,
        "matched_concepts": concepts,
        "strategy": "NLP-based",
    }
