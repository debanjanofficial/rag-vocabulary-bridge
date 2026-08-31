"""Provenance tracking and citation resolver using the Terminology Knowledge Graph.

Enables tracking of technical claims and standardized terminology back to
exact source passages, PDFs, and document pages.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.tkg.graph import TerminologyKG
from src.tkg.schema import EntityType, RelationType


class ProvenanceTracker:
    """Resolves provenance links between domain concepts and corpus passages."""

    def __init__(self, kg: TerminologyKG) -> None:
        """Initialize tracker with loaded Terminology Knowledge Graph."""
        self.kg = kg

    def get_provenance_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Retrieve all passage chunks directly linked to this entity node."""
        node = self.kg.get_node(node_id)
        if not node:
            return []

        provenance_list: list[dict[str, Any]] = []
        for target_node, edge in self.kg.get_out_edges(
            node_id, relation=RelationType.GROUNDED_IN
        ):
            if target_node.entity_type == EntityType.PASSAGE:
                provenance_list.append(target_node.properties)

        # Also check incoming edges if passages point to concepts
        for source_node, edge in self.kg.get_in_edges(
            node_id, relation=RelationType.GROUNDED_IN
        ):
            if source_node.entity_type == EntityType.PASSAGE:
                provenance_list.append(source_node.properties)

        return provenance_list

    def get_standards_for_product(self, product_name: str) -> list[str]:
        """List all technical standards governed by or linked to a product."""
        p_node = self.kg.find_node(product_name)
        if not p_node:
            return []

        standards: set[str] = set()
        for target_node, edge in self.kg.get_out_edges(
            p_node.id, relation=RelationType.GOVERNED_BY_STANDARD
        ):
            if target_node.entity_type == EntityType.STANDARD:
                standards.add(target_node.label)

        return sorted(list(standards))

    def verify_chunk_citations(
        self,
        query: str,
        retrieved_chunk_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Check whether retrieved chunk IDs contain ground-truth graph provenance."""
        q_lower = query.lower()
        matched_standards: set[str] = set()
        for std_node in self.kg.get_nodes_by_type(EntityType.STANDARD):
            if std_node.label.lower() in q_lower:
                matched_standards.add(std_node.id)

        grounded_chunks: set[str] = set()
        for std_id in matched_standards:
            for p in self.get_provenance_for_node(std_id):
                if p.get("chunk_id"):
                    grounded_chunks.add(p["chunk_id"])

        overlap = set(retrieved_chunk_ids) & grounded_chunks
        return {
            "query_standards": [
                self.kg.get_node(sid).label for sid in matched_standards if self.kg.get_node(sid)
            ],
            "grounded_chunk_ids": sorted(list(grounded_chunks)),
            "retrieved_overlap": sorted(list(overlap)),
            "is_provenance_aligned": len(overlap) > 0 if grounded_chunks else True,
        }
