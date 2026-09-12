"""In-memory Multi-Relational Terminology Knowledge Graph based on NetworkX.

Provides indexed graph queries, multi-hop relation traversals, and JSON
serialization.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import networkx as nx

from src.tkg.schema import EntityType, KGEdge, KGNode, RelationType


class TerminologyKG:
    """Multi-relational Knowledge Graph for domain terminology and provenance."""

    def __init__(self) -> None:
        """Initialize empty NetworkX MultiDiGraph and lookup indices."""
        self._graph = nx.MultiDiGraph()
        self._nodes: dict[str, KGNode] = {}
        self._label_index: dict[str, str] = {}
        self._alias_index: dict[str, str] = {}
        self._type_index: dict[EntityType, set[str]] = defaultdict(set)

    def add_node(self, node: KGNode) -> None:
        """Add or update an entity node and update lookup indices."""
        self._nodes[node.id] = node
        self._graph.add_node(
            node.id,
            label=node.label,
            entity_type=node.entity_type.value,
            definition=node.definition,
            properties=node.properties,
        )
        self._label_index[node.label.strip().lower()] = node.id
        for alias in node.aliases:
            a_clean = alias.strip().lower()
            if a_clean:
                self._alias_index[a_clean] = node.id
        self._type_index[node.entity_type].add(node.id)

    def add_edge(self, edge: KGEdge) -> None:
        """Add a directed relational edge between two existing nodes."""
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node '{edge.source_id}' not in graph.")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node '{edge.target_id}' not in graph.")

        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.relation.value,
            relation=edge.relation.value,
            weight=edge.weight,
            provenance=edge.provenance,
            properties=edge.properties,
        )

    def add_constraint_rule(
        self,
        product_label: str,
        constraint_label: str,
        rule_text: str,
        relation: RelationType = RelationType.FORBIDS_AGENT,
        provenance: str = "Human Expert Feedback",
    ) -> tuple[KGNode, KGEdge]:
        """Add a negative constraint node and link it to a product."""
        import re

        prod_node = self.find_node(product_label)
        if not prod_node:
            p_id = "prod_" + re.sub(r"[^a-z0-9]+", "_", product_label.lower()).strip("_")
            prod_node = KGNode(
                id=p_id,
                label=product_label,
                entity_type=EntityType.PRODUCT_FAMILY,
                definition=f"Product family: {product_label}",
            )
            self.add_node(prod_node)

        c_id = "const_" + re.sub(r"[^a-z0-9]+", "_", constraint_label.lower()).strip("_")
        const_node = self.get_node(c_id)
        if not const_node:
            const_node = KGNode(
                id=c_id,
                label=constraint_label,
                entity_type=EntityType.CONSTRAINT,
                definition=rule_text,
                properties={"rule_text": rule_text, "source": provenance},
            )
            self.add_node(const_node)

        edge = KGEdge(
            source_id=prod_node.id,
            target_id=const_node.id,
            relation=relation,
            provenance=provenance,
            properties={"rule_text": rule_text},
        )
        self.add_edge(edge)
        return const_node, edge

    def get_node(self, node_id: str) -> KGNode | None:
        """Retrieve node by unique ID."""
        return self._nodes.get(node_id)

    def find_node(self, text: str) -> KGNode | None:
        """Find node by matching exact label or registered alias."""
        t_clean = (text or "").strip().lower()
        if not t_clean:
            return None
        if t_clean in self._label_index:
            return self._nodes[self._label_index[t_clean]]
        if t_clean in self._alias_index:
            return self._nodes[self._alias_index[t_clean]]
        return None

    def get_nodes_by_type(self, entity_type: EntityType) -> list[KGNode]:
        """Return all nodes of a specific EntityType."""
        node_ids = self._type_index.get(entity_type, set())
        return [self._nodes[nid] for nid in node_ids]

    def get_out_edges(
        self,
        node_id: str,
        relation: RelationType | None = None,
    ) -> list[tuple[KGNode, KGEdge]]:
        """Retrieve outgoing edges and target nodes for a given node ID."""
        if node_id not in self._graph:
            return []

        results: list[tuple[KGNode, KGEdge]] = []
        for _, target_id, edge_key, data in self._graph.out_edges(
            node_id, keys=True, data=True
        ):
            edge_rel = RelationType(data.get("relation", edge_key))
            if relation is not None and edge_rel != relation:
                continue

            target_node = self._nodes[target_id]
            edge = KGEdge(
                source_id=node_id,
                target_id=target_id,
                relation=edge_rel,
                weight=float(data.get("weight", 1.0)),
                provenance=data.get("provenance", ""),
                properties=data.get("properties", {}),
            )
            results.append((target_node, edge))

        return results

    def get_in_edges(
        self,
        node_id: str,
        relation: RelationType | None = None,
    ) -> list[tuple[KGNode, KGEdge]]:
        """Retrieve incoming edges and source nodes for a given node ID."""
        if node_id not in self._graph:
            return []

        results: list[tuple[KGNode, KGEdge]] = []
        for source_id, _, edge_key, data in self._graph.in_edges(
            node_id, keys=True, data=True
        ):
            edge_rel = RelationType(data.get("relation", edge_key))
            if relation is not None and edge_rel != relation:
                continue

            source_node = self._nodes[source_id]
            edge = KGEdge(
                source_id=source_id,
                target_id=node_id,
                relation=edge_rel,
                weight=float(data.get("weight", 1.0)),
                provenance=data.get("provenance", ""),
                properties=data.get("properties", {}),
            )
            results.append((source_node, edge))

        return results

    def node_count(self) -> int:
        """Return total number of nodes in graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return total number of relational edges in graph."""
        return self._graph.number_of_edges()

    def summary(self) -> dict[str, Any]:
        """Return count breakdowns across entity types and relation types."""
        type_counts = {
            k.value: len(v) for k, v in self._type_index.items() if v
        }
        rel_counts: dict[str, int] = defaultdict(int)
        for _, _, data in self._graph.edges(data=True):
            rel_counts[data.get("relation", "unknown")] += 1

        return {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
            "entity_types": type_counts,
            "relation_types": dict(rel_counts),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete knowledge graph to dictionary format."""
        nodes_data = [n.to_dict() for n in self._nodes.values()]
        edges_data = []
        for u, v, _, data in self._graph.edges(keys=True, data=True):
            edges_data.append({
                "source_id": u,
                "target_id": v,
                "relation": data.get("relation", ""),
                "weight": data.get("weight", 1.0),
                "provenance": data.get("provenance", ""),
                "properties": data.get("properties", {}),
            })
        return {
            "nodes": nodes_data,
            "edges": edges_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerminologyKG:
        """Construct knowledge graph instance from serialized dictionary."""
        kg = cls()
        for nd in data.get("nodes", []):
            kg.add_node(KGNode.from_dict(nd))
        for ed in data.get("edges", []):
            kg.add_edge(KGEdge.from_dict(ed))
        return kg

    def save_json(self, filepath: str) -> None:
        """Save serialized graph to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> TerminologyKG:
        """Load knowledge graph from JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
