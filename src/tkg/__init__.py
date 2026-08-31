"""Terminology Knowledge Graph (TKG) package.

Provides multi-relational graph schemas, automated construction, constrained
subgraph traversal, and provenance tracking.
"""

from __future__ import annotations

import os
from functools import lru_cache

from config import DATA_DIR
from src.tkg.builder import build_terminology_kg
from src.tkg.graph import TerminologyKG
from src.tkg.provenance import ProvenanceTracker
from src.tkg.schema import (
    EntityType,
    KGEdge,
    KGNode,
    RelationType,
    SubgraphResult,
)
from src.tkg.traversal import SubgraphExtractor

TKG_JSON_PATH = os.path.join(DATA_DIR, "terminology_graph.json")
_default_tkg: TerminologyKG | None = None


@lru_cache(maxsize=1)
def get_tkg(graph_path: str = TKG_JSON_PATH) -> TerminologyKG:
    """Return singleton cached Terminology Knowledge Graph instance."""
    global _default_tkg
    if _default_tkg is None:
        if os.path.exists(graph_path):
            _default_tkg = TerminologyKG.load_json(graph_path)
        else:
            _default_tkg = build_terminology_kg()
            _default_tkg.save_json(graph_path)
    return _default_tkg


__all__ = [
    "EntityType",
    "KGEdge",
    "KGNode",
    "ProvenanceTracker",
    "RelationType",
    "SubgraphExtractor",
    "SubgraphResult",
    "TerminologyKG",
    "build_terminology_kg",
    "get_tkg",
]
