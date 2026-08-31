"""Builder pipeline constructing the multi-relational Terminology Knowledge Graph.

Integrates:
  1) Controlled vocabulary concepts and flat dictionary maps
  2) Product family hierarchies and aliases
  3) Corpus chunks, standards extraction, and passage provenance
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from config import CORPUS_CHUNKS, PRODUCT_LINES, TERMINOLOGY
from src.tkg.graph import TerminologyKG
from src.tkg.schema import EntityType, KGEdge, KGNode, RelationType

_STANDARD_PATTERNS = [
    re.compile(r"\b(DIN\s+(?:EN\s+ISO\s+|EN\s+|ISO\s+)?\d+(?:-[A-Z0-9]+)?)\b", re.IGNORECASE),
    re.compile(r"\b(ASTM\s+[A-Z0-9\-]+)\b", re.IGNORECASE),
    re.compile(r"\b(ISO\s+\d+(?:-\d+)?)\b", re.IGNORECASE),
]

_PROCESS_DEFINITIONS = {
    "proc_cutting": {
        "label": "Cutting & Machining",
        "keywords": ["cut", "cutting", "saw", "sawing", "milling", "machining", "blade", "sacrificial panel"],
    },
    "proc_pressing": {
        "label": "Pressing & Balancing",
        "keywords": ["press", "pressing", "balance", "balancing", "bend", "bending", "warp", "warpage", "substrate", "mdf"],
    },
    "proc_installation": {
        "label": "Installation & Mounting",
        "keywords": ["install", "installation", "mount", "mounting", "stud", "bracket", "screw", "fasten", "shelf"],
    },
    "proc_drilling": {
        "label": "Drilling & Fastening",
        "keywords": ["drill", "drilling", "pilot", "hole", "carbide", "screw", "rivet"],
    },
    "proc_cleaning": {
        "label": "Cleaning & Maintenance",
        "keywords": ["clean", "cleaning", "care", "maintenance", "detergent", "microfiber", "scratch"],
    },
    "proc_fire_safety": {
        "label": "Fire Behavior & Safety",
        "keywords": ["burn", "burning", "fire", "flammability", "flame", "smoke", "combustion", "din 4102", "astm e84"],
    },
    "proc_storage": {
        "label": "Storage & Acclimation",
        "keywords": ["storage", "store", "acclimation", "temperature", "humidity", "package", "unpacking"],
    },
    "proc_tolerances": {
        "label": "Tolerances & Dimensions",
        "keywords": ["tolerance", "tolerances", "variation", "deviation", "dimension", "dimensions", "length variation", "din 324"],
    },
}



def _slugify(text: str) -> str:
    """Convert arbitrary label string to a clean identifier slug."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return clean or "unknown"


def _infer_process_ids(text: str) -> list[str]:
    """Detect associated process IDs based on keyword presence."""
    t_lower = text.lower()
    matched = []
    for proc_id, p_data in _PROCESS_DEFINITIONS.items():
        for kw in p_data["keywords"]:
            if re.search(rf"\b{re.escape(kw)}\b", t_lower):
                matched.append(proc_id)
                break
    return matched


def build_terminology_kg(
    terminology_path: str = TERMINOLOGY,
    product_lines_path: str = PRODUCT_LINES,
    corpus_chunks_path: str = CORPUS_CHUNKS,
) -> TerminologyKG:
    """Construct complete Terminology Knowledge Graph from repository sources."""
    kg = TerminologyKG()

    # 1. Add Process Nodes
    for proc_id, p_data in _PROCESS_DEFINITIONS.items():
        proc_node = KGNode(
            id=proc_id,
            label=p_data["label"],
            entity_type=EntityType.PROCESS,
            definition=f"Industrial procedure category for {p_data['label']}.",
            aliases=p_data["keywords"],
        )
        kg.add_node(proc_node)

    # 2. Add Product Family Nodes from product_lines.json
    prod_family_nodes: dict[str, str] = {}  # prod_name_lower -> node_id
    if os.path.exists(product_lines_path):
        with open(product_lines_path, encoding="utf-8") as f:
            pl_data = json.load(f)
        for key, entry in pl_data.items():
            if key.strip().lower() == "general":
                continue
            preferred = (entry.get("preferred_term") or key).strip()
            node_id = f"prod_{_slugify(preferred)}"
            synonyms = entry.get("synonyms_colloquial") or []
            prod_node = KGNode(
                id=node_id,
                label=preferred,
                entity_type=EntityType.PRODUCT_FAMILY,
                definition=entry.get("description", f"REHAU product line: {preferred}"),
                aliases=synonyms,
                properties={"category": entry.get("category", "")},
            )
            kg.add_node(prod_node)
            prod_family_nodes[preferred.lower()] = node_id
            for syn in synonyms:
                prod_family_nodes[syn.lower()] = node_id

    # 3. Add Concepts and Colloquial Mappings from terminology.json
    if os.path.exists(terminology_path):
        with open(terminology_path, encoding="utf-8") as f:
            term_data = json.load(f)

        concepts = term_data.get("concepts", [])
        flat_map = term_data.get("flat_map", {})

        # Map preferred_term -> concept_id for cross-referencing
        pref_to_node: dict[str, str] = {}

        for c in concepts:
            preferred = c.get("preferred_term", "").strip()
            if not preferred:
                continue
            c_id = c.get("concept_id") or f"tech_{_slugify(preferred)}"
            pref_to_node[preferred.lower()] = c_id

            tech_node = KGNode(
                id=c_id,
                label=preferred,
                entity_type=EntityType.TECHNICAL_TERM,
                definition=c.get("definition", ""),
                aliases=c.get("acronyms", []) + c.get("synonyms_colloquial", []),
                properties={"characteristics": c.get("characteristics", [])},
            )
            kg.add_node(tech_node)

            # Link to Process
            proc_ids = _infer_process_ids(preferred + " " + c.get("definition", ""))
            for pid in proc_ids:
                kg.add_edge(KGEdge(
                    source_id=c_id,
                    target_id=pid,
                    relation=RelationType.INVOLVES_PROCESS,
                    weight=0.9,
                ))

            # Link to Product Family if concept mentions product
            for prod_name, p_node_id in prod_family_nodes.items():
                if prod_name in preferred.lower():
                    kg.add_edge(KGEdge(
                        source_id=c_id,
                        target_id=p_node_id,
                        relation=RelationType.BELONGS_TO_PRODUCT,
                        weight=1.0,
                    ))
                    break

            # Add Colloquial Synonyms as nodes & edges
            for syn in c.get("synonyms_colloquial", []):
                s_clean = syn.strip()
                if not s_clean:
                    continue
                col_id = f"col_{_slugify(s_clean)}"
                if not kg.get_node(col_id):
                    kg.add_node(KGNode(
                        id=col_id,
                        label=s_clean,
                        entity_type=EntityType.COLLOQUIAL_TERM,
                        definition=f"Everyday expression mapping to {preferred}.",
                    ))
                kg.add_edge(KGEdge(
                    source_id=col_id,
                    target_id=c_id,
                    relation=RelationType.MAPS_TO_PREFERRED,
                    weight=0.95,
                ))

        # Ingest remaining flat_map pairs
        for lay_term, formal_term in flat_map.items():
            lay_clean = lay_term.strip()
            formal_clean = formal_term.strip()
            if not lay_clean or not formal_clean:
                continue

            target_node_id = pref_to_node.get(formal_clean.lower())
            if not target_node_id:
                # Check if formal term is a product family
                if formal_clean.lower() in prod_family_nodes:
                    target_node_id = prod_family_nodes[formal_clean.lower()]
                else:
                    target_node_id = f"tech_{_slugify(formal_clean)}"
                    if not kg.get_node(target_node_id):
                        kg.add_node(KGNode(
                            id=target_node_id,
                            label=formal_clean,
                            entity_type=EntityType.TECHNICAL_TERM,
                            definition=f"Controlled vocabulary term: {formal_clean}",
                        ))
                    pref_to_node[formal_clean.lower()] = target_node_id

            col_id = f"col_{_slugify(lay_clean)}"
            if not kg.get_node(col_id):
                kg.add_node(KGNode(
                    id=col_id,
                    label=lay_clean,
                    entity_type=EntityType.COLLOQUIAL_TERM,
                    definition=f"Everyday expression mapping to {formal_clean}.",
                ))

            # Add edge if not already present
            existing_targets = {
                target.id for target, _ in kg.get_out_edges(
                    col_id, relation=RelationType.MAPS_TO_PREFERRED
                )
            }
            if target_node_id not in existing_targets:
                kg.add_edge(KGEdge(
                    source_id=col_id,
                    target_id=target_node_id,
                    relation=RelationType.MAPS_TO_PREFERRED,
                    weight=1.0,
                ))

    # 4. Ingest Corpus Chunks and Extract Standards Provenance
    if os.path.exists(corpus_chunks_path):
        with open(corpus_chunks_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                chunk = json.loads(line_str)
                chunk_id = chunk.get("chunk_id", "")
                text = chunk.get("text", "")
                product = chunk.get("product", "")
                source_pdf = chunk.get("source_pdf", "")

                # Create Passage Node
                passage_node_id = f"chunk_{chunk_id}"
                kg.add_node(KGNode(
                    id=passage_node_id,
                    label=chunk_id,
                    entity_type=EntityType.PASSAGE,
                    definition=text[:200] + "...",
                    properties={
                        "chunk_id": chunk_id,
                        "source_pdf": source_pdf,
                        "product": product,
                        "doc_type": chunk.get("doc_type", ""),
                        "page_start": chunk.get("page_start", 0),
                        "page_end": chunk.get("page_end", 0),
                    },
                ))

                # Link Passage to Product Family
                if product.lower() in prod_family_nodes:
                    p_node_id = prod_family_nodes[product.lower()]
                    kg.add_edge(KGEdge(
                        source_id=passage_node_id,
                        target_id=p_node_id,
                        relation=RelationType.BELONGS_TO_PRODUCT,
                        weight=1.0,
                    ))

                # Extract and Link Standards
                for pat in _STANDARD_PATTERNS:
                    for match in pat.finditer(text):
                        std_name = match.group(1).strip()
                        std_id = f"std_{_slugify(std_name)}"
                        if not kg.get_node(std_id):
                            kg.add_node(KGNode(
                                id=std_id,
                                label=std_name,
                                entity_type=EntityType.STANDARD,
                                definition=f"Technical specification standard: {std_name}",
                            ))
                        kg.add_edge(KGEdge(
                            source_id=std_id,
                            target_id=passage_node_id,
                            relation=RelationType.GROUNDED_IN,
                            provenance=source_pdf,
                            weight=1.0,
                        ))
                        # Connect standard to product if chunk has product
                        if product.lower() in prod_family_nodes:
                            p_node_id = prod_family_nodes[product.lower()]
                            kg.add_edge(KGEdge(
                                source_id=p_node_id,
                                target_id=std_id,
                                relation=RelationType.GOVERNED_BY_STANDARD,
                                weight=0.9,
                            ))

    return kg
