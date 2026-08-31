"""Constrained Subgraph Traversal and Multi-Hop Query Expansion.

Given a user query, identifies entry entity nodes, performs bounded k-hop
relational graph traversal, and constructs provenance-grounded query
expansions.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.tkg.graph import TerminologyKG
from src.tkg.schema import EntityType, KGEdge, KGNode, RelationType, SubgraphResult


class SubgraphExtractor:
    """Extracts constrained subgraphs and multi-hop expansions for queries."""

    def __init__(self, kg: TerminologyKG) -> None:
        """Initialize extractor with loaded Terminology Knowledge Graph."""
        self.kg = kg

    def find_entry_nodes(self, query: str) -> list[KGNode]:
        """Identify initial entity nodes directly mentioned in the query."""
        q_norm = re.sub(r"[\s\-]+", " ", query.lower()).strip()
        matched: list[KGNode] = []
        seen_ids: set[str] = set()

        candidates = (
            self.kg.get_nodes_by_type(EntityType.COLLOQUIAL_TERM)
            + self.kg.get_nodes_by_type(EntityType.TECHNICAL_TERM)
            + self.kg.get_nodes_by_type(EntityType.PRODUCT_FAMILY)
        )

        # Sort candidates by length descending to match longest specific phrases first
        sorted_candidates = sorted(
            candidates,
            key=lambda n: max(len(n.label), max([len(a) for a in n.aliases] or [0])),
            reverse=True,
        )

        for node in sorted_candidates:
            # Check normalized label
            lbl_norm = re.sub(r"[\s\-]+", " ", node.label.lower()).strip()
            if len(lbl_norm) >= 3 and re.search(rf"(?<!\w){re.escape(lbl_norm)}(?!\w)", q_norm):
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    matched.append(node)
                continue

            # Check individual phrases / stems if label is multi-word
            if len(lbl_norm) >= 6 and " " in lbl_norm:
                tokens = lbl_norm.split()
                # Check if all key tokens (non-stopwords) appear in query
                key_tokens = [t for t in tokens if len(t) >= 3 and t not in ("for", "the", "and", "with")]
                if len(key_tokens) >= 2 and all(re.search(rf"(?<!\w){re.escape(t.rstrip('s'))}(?:s)?(?!\w)", q_norm) for t in key_tokens):
                    if node.id not in seen_ids:
                        seen_ids.add(node.id)
                        matched.append(node)
                    continue

            # Check aliases
            for alias in node.aliases:
                a_norm = re.sub(r"[\s\-]+", " ", alias.lower()).strip()
                if len(a_norm) >= 3 and re.search(rf"(?<!\w){re.escape(a_norm)}(?!\w)", q_norm):
                    if node.id not in seen_ids:
                        seen_ids.add(node.id)
                        matched.append(node)
                    break

                if len(a_norm) >= 6 and " " in a_norm:
                    tokens = a_norm.split()
                    key_tokens = [
                        t for t in tokens
                        if len(t) >= 3 and t not in (
                            "for", "the", "and", "with", "panel", "panels",
                            "material", "materials", "board", "boards", "parts"
                        )
                    ]
                    if len(key_tokens) >= 2 and all(
                        re.search(rf"(?<!\w){re.escape(t.rstrip('s'))}(?:s)?(?!\w)", q_norm)
                        for t in key_tokens
                    ):
                        if node.id not in seen_ids:
                            seen_ids.add(node.id)
                            matched.append(node)
                        break

        return matched



    def extract_subgraph(
        self,
        query: str,
        max_hops: int = 2,
    ) -> SubgraphResult:
        """Extract multi-hop subgraph and generate expanded retrieval query."""
        entry_nodes = self.find_entry_nodes(query)
        q_lower = query.lower()

        visited_nodes: dict[str, KGNode] = {n.id: n for n in entry_nodes}
        collected_edges: list[KGEdge] = []

        preferred_terms: set[str] = set()
        product_families: set[str] = set()
        processes: set[str] = set()
        standards: set[str] = set()
        provenance_chunks: set[str] = set()

        # Process detection for query context constraint
        query_proc_nodes = []
        for proc_node in self.kg.get_nodes_by_type(EntityType.PROCESS):
            for alias in proc_node.aliases:
                if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", q_lower):
                    query_proc_nodes.append(proc_node)
                    processes.add(proc_node.label)
                    visited_nodes[proc_node.id] = proc_node
                    break

        # Breadth-first traversal starting from entry nodes
        current_level_ids = [n.id for n in entry_nodes]

        for _ in range(max_hops):
            next_level_ids = []
            for nid in current_level_ids:
                # Get outgoing relational edges
                for target_node, edge in self.kg.get_out_edges(nid):
                    collected_edges.append(edge)
                    if target_node.id not in visited_nodes:
                        visited_nodes[target_node.id] = target_node
                        next_level_ids.append(target_node.id)

                    # Categorize target node
                    if target_node.entity_type == EntityType.TECHNICAL_TERM:
                        preferred_terms.add(target_node.label)
                    elif target_node.entity_type == EntityType.PRODUCT_FAMILY:
                        product_families.add(target_node.label)
                    elif target_node.entity_type == EntityType.PROCESS:
                        processes.add(target_node.label)
                    elif target_node.entity_type == EntityType.STANDARD:
                        standards.add(target_node.label)
                    elif target_node.entity_type == EntityType.PASSAGE:
                        chunk_id = target_node.properties.get("chunk_id", target_node.label)
                        provenance_chunks.add(chunk_id)

            current_level_ids = next_level_ids
            if not current_level_ids:
                break

        # Also search for standards linked to active product families
        for p_label in list(product_families):
            p_node = self.kg.find_node(p_label)
            if p_node:
                for target_node, edge in self.kg.get_out_edges(
                    p_node.id, relation=RelationType.GOVERNED_BY_STANDARD
                ):
                    collected_edges.append(edge)
                    visited_nodes[target_node.id] = target_node
                    standards.add(target_node.label)

        # Construct multi-hop expanded query
        expansion_terms = []
        for pref in sorted(preferred_terms):
            if pref.lower() not in q_lower:
                expansion_terms.append(pref)

        for prod in sorted(product_families):
            if prod.lower() not in q_lower and prod not in expansion_terms:
                expansion_terms.append(prod)

        # Prioritize process-specific standards
        prioritized_standards = []
        if "Fire Behavior & Safety" in processes:
            for s in standards:
                if "4102" in s or "e84" in s.lower():
                    prioritized_standards.append(s)
        if "Tolerances & Dimensions" in processes:
            for s in standards:
                if "324" in s:
                    prioritized_standards.append(s)
        if "Pressing & Balancing" in processes:
            for s in standards:
                if "178" in s or "iso" in s.lower():
                    prioritized_standards.append(s)

        remaining_standards = [s for s in standards if s not in prioritized_standards]
        final_standards = (prioritized_standards + sorted(remaining_standards))[:2]

        for std in final_standards:
            if std.lower() not in q_lower:
                expansion_terms.append(std)

        if expansion_terms:
            expansion_suffix = " (" + " ".join(expansion_terms[:4]) + ")"
            expanded_query = query.strip() + expansion_suffix
        else:
            expanded_query = query.strip()

        return SubgraphResult(
            matched_nodes=entry_nodes,
            subgraph_nodes=list(visited_nodes.values()),
            subgraph_edges=collected_edges,
            preferred_terms=sorted(list(preferred_terms)),
            product_families=sorted(list(product_families)),
            processes=sorted(list(processes)),
            standards=sorted(list(standards)),
            provenance_chunk_ids=sorted(list(provenance_chunks)),
            expansion_query=expanded_query,
        )

