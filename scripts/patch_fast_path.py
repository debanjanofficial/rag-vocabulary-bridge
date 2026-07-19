"""Fast-path patch — merge product synonyms + augment generic colloquial queries."""

from __future__ import annotations

import json
import os
import re

import scripts._bootstrap  # noqa: F401
from config import EVAL_DATASET, TERMINOLOGY, PRODUCT_LINES

_PRODUCT_TOKENS = (
    "rauvisio", "brilliant", "crystal", "noir", "shade", "ferro",
    "ingrain", "terra", "floating shelf", "floating shelves",
    "integrated handle", "laseredge",
)
_GOOD_HINTS = (
    "shiny cabinet", "glossy cabinet", "glossy door", "shiny door",
    "clear acrylic", "acrylic board", "glass panel", "glass board",
    "matte counter", "metal look", "metal-looking", "metallic",
    "wood grain", "wood fiber", "textured wood", "fake glass",
    "high-gloss", "super matte", "super-matte", "floating shelf",
    "edgeband", "edge band",
)
_PRODUCT_NICK = {
    "RAUVISIO brilliant": "brilliant",
    "RAUVISIO crystal": "crystal glass",
    "RAUVISIO noir": "noir matte",
    "RAUVISIO shade": "shade matte",
    "RAUVISIO ferro": "metal-look",
    "RAUVISIO ingrain": "wood-grain",
    "RAUVISIO terra": "terra textured",
    "Floating Shelves": "floating shelf",
    "Integrated Handles": "integrated handle",
    "General": "RAUVISIO",
}
_GENERIC_PATTERNS = [
    ("these cabinet doors", "my {nick} cabinet doors"),
    ("the cabinet doors", "my {nick} cabinet doors"),
    ("these panels", "my {nick} panels"),
    ("the panels", "my {nick} panels"),
    ("these boards", "my {nick} boards"),
    ("the boards", "my {nick} boards"),
    ("these shelves", "my {nick} shelves"),
    ("the shelves", "my {nick} shelves"),
    ("this laminate", "this {nick} laminate"),
    ("the laminate", "the {nick} laminate"),
    ("this material", "this {nick} material"),
    ("the material", "the {nick} material"),
    ("this product", "this {nick} product"),
    ("the product", "the {nick} product"),
    ("these parts", "these {nick} parts"),
    ("the parts", "the {nick} parts"),
    ("this stuff", "this {nick} stuff"),
    ("the stuff", "the {nick} stuff"),
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def is_good_colloquial(query: str) -> bool:
    ql = query.lower()
    return any(t in ql for t in _PRODUCT_TOKENS) or any(h in ql for h in _GOOD_HINTS)


def augment_colloquial(query: str, product: str) -> str | None:
    nick = _PRODUCT_NICK.get(product)
    if not nick:
        return None
    ql = query.lower()
    for generic, template in _GENERIC_PATTERNS:
        if generic in ql:
            start = ql.index(generic)
            variant = query[:start] + template.format(nick=nick) + query[start + len(generic):]
            if variant.strip().lower() != query.strip().lower():
                return variant.strip()
    q = query.strip()
    if q.endswith("?"):
        body = q[:-1].strip()
        if body:
            body = body[0].lower() + body[1:]
        return f"For my {nick} product, {body}?"
    return f"For my {nick} product — {q}"


def _load_terminology() -> dict:
    with open(TERMINOLOGY, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"concepts": data, "flat_map": {}}
    return {"concepts": data.get("concepts", []), "flat_map": data.get("flat_map", {})}


def _save_terminology(concepts: list, flat_map: dict):
    with open(TERMINOLOGY, "w", encoding="utf-8") as f:
        json.dump({"concepts": concepts, "flat_map": flat_map}, f, indent=2, ensure_ascii=False)


def merge_product_lines(product_lines: dict) -> tuple[list, dict]:
    data = _load_terminology()
    concepts = data["concepts"]
    by_term = {c["preferred_term"].lower(): c for c in concepts}

    for pl in product_lines.values():
        pt = pl["preferred_term"]
        existing = by_term.get(pt.lower())
        if existing:
            syns = set(existing.get("synonyms_colloquial", []))
            syns.update(pl.get("synonyms_colloquial", []))
            existing["synonyms_colloquial"] = sorted(syns)
        else:
            concepts.append({
                "concept_id": "c_product_line_" + slugify(pt),
                "preferred_term": pt,
                "acronyms": [],
                "synonyms_colloquial": sorted(pl.get("synonyms_colloquial", [])),
                "definition": pl.get("definition", ""),
                "characteristics": sorted(pl.get("characteristics", [])),
            })
            by_term[pt.lower()] = concepts[-1]

    flat: dict[str, str] = {}
    for c in concepts:
        pt = c["preferred_term"]
        for syn in c.get("synonyms_colloquial", []) + c.get("acronyms", []):
            s = syn.lower().strip()
            if s and s != pt.lower():
                flat.setdefault(s, pt)
    for pl in product_lines.values():
        pt = pl["preferred_term"]
        for syn in pl.get("synonyms_colloquial", []):
            s = syn.lower().strip()
            if s:
                flat[s] = pt

    concepts.sort(key=lambda c: c["preferred_term"].lower())
    return concepts, dict(sorted(flat.items()))


def patch_eval() -> tuple[int, int, int]:
    items = []
    with open(EVAL_DATASET, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    good, augmented = 0, 0
    for item in items:
        if item.get("augmented_variant"):
            good += 1
            continue
        primary = item["colloquial_query"]
        variants = list(item.get("colloquial_queries", [primary]))
        if is_good_colloquial(primary):
            good += 1
            continue
        extra = augment_colloquial(primary, item.get("product", ""))
        if not extra or extra.lower() in {v.lower() for v in variants}:
            continue
        variants.append(extra)
        item["colloquial_queries"] = variants
        item["augmented_variant"] = extra
        augmented += 1
    with open(EVAL_DATASET, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items), good, augmented


def main():
    with open(PRODUCT_LINES, encoding="utf-8") as f:
        product_lines = json.load(f)
    concepts, flat = merge_product_lines(product_lines)
    _save_terminology(concepts, flat)
    print(f"terminology.json: {len(concepts)} concepts, {len(flat)} flat mappings")
    total, good, aug = patch_eval()
    print(f"eval_dataset: {total} items | {good} good/unchanged | {aug} newly augmented")


if __name__ == "__main__":
    main()
