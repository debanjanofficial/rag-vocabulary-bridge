"""LLM-based query mapping — Self-Query (semantic query + metadata filters).

The LLM parses a colloquial question into:
  - intent (install / fire / process / clean / specs / other)
  - semantic_query: procedure-oriented search text
  - product / doc_type: optional metadata filters (must match corpus values)

When product is set, callers retrieve with a HARD product filter, then
intent-aware rerank (prefer Technical Guide vs SpecSheets). Abstain → baseline.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from config import PRODUCT_LINES
from src.llm import chat_json

ALLOWED_PRODUCTS = (
    "RAUVISIO brilliant",
    "RAUVISIO crystal",
    "RAUVISIO ferro",
    "RAUVISIO ingrain",
    "RAUVISIO noir",
    "RAUVISIO shade",
    "RAUVISIO terra",
    "Floating Shelves",
    "Integrated Handles",
    "General",
)

ALLOWED_DOC_TYPES = (
    "Technical Documentation",
    "Marketing Materials",
    "Product Information",
    "Root",
    "General",
)

ALLOWED_INTENTS = frozenset({
    "install", "fire", "process", "clean", "safety", "specs", "other",
})

# How-to / procedure → prefer Technical Guide.
# Catalog / models → prefer SpecSheets.
# other → no source bias.
_GUIDE_INTENTS = frozenset({"install", "fire", "process", "clean", "safety"})
_SPEC_INTENTS = frozenset({"specs"})

_INTENT_QUERIES = {
    "install": "installation mounting fastening drilling hinges procedure technical",
    "fire": "fire behavior DIN 4102 flammability flame spread smoke classification",
    "process": "cutting sawing pressing machining milling processing tools",
    "clean": "cleaning care maintenance approved cleaners detergent",
    "safety": "safety warnings protective equipment dust mask handling",
    "specs": "approved manufacturers product models specifications dimensions SKU",
}

_PRODUCT_ALIASES: dict[str, str] = {
    "rehau floating shelves": "Floating Shelves",
    "floating shelf": "Floating Shelves",
    "floating shelves": "Floating Shelves",
    "rehau integrated handles": "Integrated Handles",
    "integrated handles": "Integrated Handles",
    "integrated handle": "Integrated Handles",
    "rauvisio house": "General",
    "rauvisio": "General",
}


def _load_product_hints() -> str:
    try:
        with open(PRODUCT_LINES, encoding="utf-8") as f:
            lines_data = json.load(f)
    except OSError:
        return ""
    rows = []
    for key, entry in lines_data.items():
        if key.strip().lower() == "general":
            continue
        preferred = entry.get("preferred_term") or key
        syns = (entry.get("synonyms_colloquial") or [])[:4]
        if syns:
            rows.append(f'  "{preferred}" ← e.g. {", ".join(repr(s) for s in syns)}')
    return "\n".join(rows)


def _normalize_product(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if v in ALLOWED_PRODUCTS:
        return v
    alias = _PRODUCT_ALIASES.get(v.lower())
    if alias:
        return alias
    vl = v.lower()
    for allowed in ALLOWED_PRODUCTS:
        if allowed.lower() == vl:
            return allowed
    for allowed in ALLOWED_PRODUCTS:
        if allowed.lower() in vl or vl in allowed.lower():
            return allowed
    return None


def _normalize_doc_type(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if v in ALLOWED_DOC_TYPES:
        return v
    vl = v.lower()
    for allowed in ALLOWED_DOC_TYPES:
        if allowed.lower() == vl:
            return allowed
    if "tech" in vl:
        return "Technical Documentation"
    if "market" in vl:
        return "Marketing Materials"
    if "product" in vl:
        return "Product Information"
    return None


def _normalize_intent(value: str | None) -> str:
    if not value or not isinstance(value, str):
        return "other"
    v = value.strip().lower()
    if v in ALLOWED_INTENTS:
        return v
    if any(k in v for k in ("install", "mount", "hang", "put up")):
        return "install"
    if "fire" in v or "flam" in v or "smoke" in v:
        return "fire"
    if any(k in v for k in ("cut", "press", "mill", "process", "saw", "drill")):
        return "process"
    if "clean" in v or "care" in v:
        return "clean"
    if "safe" in v or "warning" in v:
        return "safety"
    if any(k in v for k in ("spec", "model", "manufacturer", "sku", "dimension")):
        return "specs"
    return "other"


def _source_preference(intent: str) -> str | None:
    if intent in _GUIDE_INTENTS:
        return "guide"
    if intent in _SPEC_INTENTS:
        return "spec"
    return None


def _strip_product_mentions(text: str) -> str:
    """Remove formal product names so search focuses on the procedure."""
    out = text
    for p in ALLOWED_PRODUCTS:
        out = re.sub(re.escape(p), " ", out, flags=re.IGNORECASE)
    out = re.sub(
        r"\b(rauvisio|rehau|brilliant|crystal|ferro|ingrain|noir|shade|terra)\b",
        " ",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s+", " ", out).strip(" ,.-")
    return out


def _refine_semantic(semantic: str, intent: str, original: str) -> str:
    """Prefer controlled procedure text for known intents."""
    if intent in _INTENT_QUERIES:
        return _INTENT_QUERIES[intent]
    cleaned = _strip_product_mentions(semantic or original)
    return cleaned if len(cleaned) >= 6 else (semantic or original)


def _infer_intent_from_query(query: str) -> str | None:
    """Lightweight fallback if LLM omits intent."""
    q = query.lower()
    if any(w in q for w in ("install", "put up", "put these", "hang", "mount")):
        return "install"
    if any(w in q for w in ("fire", "burn", "flammab", "smoke rating")):
        return "fire"
    if any(w in q for w in ("cut", "press", "saw", "mill", "drill")):
        return "process"
    if any(w in q for w in ("clean", "windex", "wipe", "care")):
        return "clean"
    if any(w in q for w in ("who makes", "manufacturer", "model", "what types", "sku")):
        return "specs"
    return None


@lru_cache(maxsize=1)
def _prompt_context() -> str:
    hints = _load_product_hints()
    products = "\n".join(f"  - {p}" for p in ALLOWED_PRODUCTS)
    doc_types = "\n".join(f"  - {d}" for d in ALLOWED_DOC_TYPES)
    return (
        f"ALLOWED PRODUCTS (copy exactly or null):\n{products}\n\n"
        f"ALLOWED DOC TYPES (copy exactly or null):\n{doc_types}\n\n"
        "ALLOWED INTENTS (copy exactly):\n"
        "  - install — mounting, hanging, putting panels/doors up\n"
        "  - fire — fire rating, flammability, smoke, DIN 4102\n"
        "  - process — cutting, pressing, machining, drilling how-to\n"
        "  - clean — cleaning and care\n"
        "  - safety — warnings, PPE, hazards\n"
        "  - specs — manufacturers, models, SKUs, sizes, approved products\n"
        "  - other — anything else\n\n"
        f"Colloquial product cues:\n{hints or '  (none)'}"
    )


def self_query(query: str) -> tuple[dict, dict]:
    """Parse query into semantic search + validated metadata filters."""
    ctx = _prompt_context()
    prompt = (
        "You are a REHAU RAUVISIO documentation search assistant.\n"
        "Parse the user's casual question for retrieval.\n\n"
        f"{ctx}\n\n"
        f'USER QUERY: "{query}"\n\n'
        "Return JSON only with these fields:\n"
        '  "intent": one of the ALLOWED INTENTS.\n'
        '  "semantic_query": string — concise technical search text for the INTENT;\n'
        "      once product is known, do NOT include product nicknames "
        '(no "shiny acrylic", "metal look", "crystal").\n'
        '  "product": string|null — EXACT ALLOWED PRODUCT, or null if unsure.\n'
        '  "doc_type": string|null — EXACT ALLOWED DOC TYPE, or null if unsure.\n'
        '  "abstain_filter": boolean — true if not confident about product.\n\n'
        "Rules:\n"
        "- Map informal nicknames to ALLOWED PRODUCTS.\n"
        "- Product disambiguation (important):\n"
        '  * shiny/glossy/high-gloss acrylic (NO glass/crystal words) → '
        '"RAUVISIO brilliant".\n'
        '  * glass, crystal glass, fake glass, glass-like, clear glass → '
        '"RAUVISIO crystal".\n'
        '  * "shiny acrylic" / "shiny acrylic panels" → ALWAYS '
        '"RAUVISIO brilliant" (not crystal).\n'
        "- If acrylic vs glass is still unclear, set abstain_filter=true "
        "(prefer no filter over a wrong product).\n"
        "- install/process/clean/fire/safety → doc_type=Technical Documentation.\n"
        "- specs (models/manufacturers) → doc_type=Technical Documentation.\n"
        "- Prefer abstain_filter=true over a wrong product.\n"
        "- Do not answer the question; only parse it."
    )

    raw = chat_json(prompt, temperature=0.2, num_predict=400) or {}
    abstain = bool(raw.get("abstain_filter"))

    intent = _normalize_intent(raw.get("intent"))
    if intent == "other":
        guessed = _infer_intent_from_query(query)
        if guessed:
            intent = guessed

    semantic = (raw.get("semantic_query") or query).strip()
    if len(semantic) < 3:
        semantic = query
    semantic = _refine_semantic(semantic, intent, query)

    product = None if abstain else _normalize_product(raw.get("product"))
    doc_type = None if abstain else _normalize_doc_type(raw.get("doc_type"))

    # Deterministic override: shiny/glossy acrylic without glass cues → brilliant
    # (LLM often confuses this with crystal).
    q_lower = query.lower()
    has_acrylic_gloss = (
        ("acrylic" in q_lower and any(w in q_lower for w in ("shiny", "glossy", "high-gloss", "high gloss")))
        or "shiny acrylic" in q_lower
        or "glossy acrylic" in q_lower
    )
    has_glass_cue = any(
        w in q_lower
        for w in ("glass", "crystal", "fake glass", "glass-like", "glass like")
    )
    if has_acrylic_gloss and not has_glass_cue:
        product = "RAUVISIO brilliant"
        abstain = False

    # Procedure questions default to technical docs when a product is set.
    if product and not doc_type and intent in _GUIDE_INTENTS | _SPEC_INTENTS:
        doc_type = "Technical Documentation"

    prefer = _source_preference(intent) if product else None

    parsed = {
        "semantic_query": semantic,
        "product": product,
        "doc_type": doc_type,
        "intent": intent,
        "prefer_source": prefer,
        "abstain_filter": abstain or not product,
    }
    debug = {
        "method": "self_query",
        "raw_llm": raw,
        "validated": parsed,
    }
    return parsed, debug


def llm_map(query: str, llm_available: bool = True, method: str = "self_query") -> dict:
    """Self-Query mapping. `method` kept for API compat (only self_query is used)."""
    del method

    if not llm_available:
        return {
            "original_query": query,
            "final_query": query,
            "semantic_query": query,
            "filters": {},
            "intent": "other",
            "prefer_source": None,
            "abstained": True,
            "retrieval_queries": [query],
            "method_used": "passthrough (no LLM)",
            "debug": {"info": "Ollama not available"},
            "strategy": "LLM-based",
        }

    try:
        parsed, debug = self_query(query)
        product = parsed.get("product")
        doc_type = parsed.get("doc_type")
        semantic = parsed.get("semantic_query") or query
        intent = parsed.get("intent") or "other"
        prefer = parsed.get("prefer_source")
        has_filter = bool(product)

        filters: dict = {}
        if product:
            filters["product"] = product
        if doc_type:
            filters["doc_type"] = doc_type

        return {
            "original_query": query,
            "final_query": semantic,
            "semantic_query": semantic,
            "filters": filters,
            "intent": intent,
            "prefer_source": prefer,
            "abstained": not has_filter,
            "retrieval_queries": [query],
            "filtered_query": semantic if has_filter else None,
            "method_used": "self_query" if has_filter else "passthrough (no filter)",
            "debug": debug,
            "strategy": "LLM-based",
        }
    except Exception as e:
        return {
            "original_query": query,
            "final_query": query,
            "semantic_query": query,
            "filters": {},
            "intent": "other",
            "prefer_source": None,
            "abstained": True,
            "retrieval_queries": [query],
            "method_used": "passthrough (error)",
            "debug": {"error": str(e)},
            "strategy": "LLM-based",
        }
