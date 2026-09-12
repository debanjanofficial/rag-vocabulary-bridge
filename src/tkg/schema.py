"""Data schema and type definitions for the Terminology Knowledge Graph (TKG).

Defines multi-relational entity types, relation types, node/edge models,
and traversal result containers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    """Discrete entity types in the multi-relational knowledge graph."""

    COLLOQUIAL_TERM = "colloquial_term"
    TECHNICAL_TERM = "technical_term"
    PRODUCT_FAMILY = "product_family"
    PROCESS = "process"
    STANDARD = "standard"
    ATTRIBUTE = "attribute"
    PASSAGE = "passage"
    CONSTRAINT = "constraint"


class RelationType(str, Enum):
    """Directed edge relationships connecting knowledge graph nodes."""

    MAPS_TO_PREFERRED = "MAPS_TO_PREFERRED"
    BELONGS_TO_PRODUCT = "BELONGS_TO_PRODUCT"
    INVOLVES_PROCESS = "INVOLVES_PROCESS"
    GOVERNED_BY_STANDARD = "GOVERNED_BY_STANDARD"
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
    GROUNDED_IN = "GROUNDED_IN"
    REQUIRES_TOOL = "REQUIRES_TOOL"
    REQUIRES_MATERIAL = "REQUIRES_MATERIAL"
    FORBIDS_PROCEDURE = "FORBIDS_PROCEDURE"
    FORBIDS_AGENT = "FORBIDS_AGENT"


@dataclass
class KGNode:
    """Individual entity node in the Terminology Knowledge Graph."""

    id: str
    label: str
    entity_type: EntityType
    definition: str = ""
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert node to a JSON-serializable dictionary."""
        data = asdict(self)
        data["entity_type"] = self.entity_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KGNode:
        """Instantiate node from a dictionary payload."""
        return cls(
            id=data["id"],
            label=data["label"],
            entity_type=EntityType(data["entity_type"]),
            definition=data.get("definition", ""),
            aliases=data.get("aliases", []),
            properties=data.get("properties", {}),
        )


@dataclass
class KGEdge:
    """Directed relational edge connecting two knowledge graph nodes."""

    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0
    provenance: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert edge to a JSON-serializable dictionary."""
        data = asdict(self)
        data["relation"] = self.relation.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KGEdge:
        """Instantiate edge from a dictionary payload."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=RelationType(data["relation"]),
            weight=float(data.get("weight", 1.0)),
            provenance=data.get("provenance", ""),
            properties=data.get("properties", {}),
        )


@dataclass
class SubgraphResult:
    """Outcome of a constrained subgraph extraction and expansion."""

    matched_nodes: list[KGNode]
    subgraph_nodes: list[KGNode]
    subgraph_edges: list[KGEdge]
    preferred_terms: list[str]
    product_families: list[str]
    processes: list[str]
    standards: list[str]
    provenance_chunk_ids: list[str]
    expansion_query: str

    def to_dict(self) -> dict[str, Any]:
        """Convert subgraph result to JSON-serializable dictionary."""
        return {
            "matched_nodes": [n.to_dict() for n in self.matched_nodes],
            "subgraph_nodes": [n.to_dict() for n in self.subgraph_nodes],
            "subgraph_edges": [e.to_dict() for e in self.subgraph_edges],
            "preferred_terms": self.preferred_terms,
            "product_families": self.product_families,
            "processes": self.processes,
            "standards": self.standards,
            "provenance_chunk_ids": self.provenance_chunk_ids,
            "expansion_query": self.expansion_query,
        }
