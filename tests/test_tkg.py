"""Unit and integration tests for the Terminology Knowledge Graph (TKG) module."""

import os
import unittest

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


class TestTKGSchemaAndGraph(unittest.TestCase):
    """Unit tests for TKG schema, graph indexing, and traversals."""

    def setUp(self):
        self.kg = TerminologyKG()
        # Add product
        self.node_prod = KGNode(
            id="prod_crystal",
            label="RAUVISIO crystal",
            entity_type=EntityType.PRODUCT_FAMILY,
            aliases=["crystal glass", "acrylic glass"],
        )
        # Add technical concept
        self.node_tech = KGNode(
            id="tech_drilling",
            label="Carbide End Mill",
            entity_type=EntityType.TECHNICAL_TERM,
            definition="Drill bit type for acrylic panels.",
        )
        # Add process
        self.node_proc = KGNode(
            id="proc_drill",
            label="Drilling & Fastening",
            entity_type=EntityType.PROCESS,
            aliases=["drill", "drilling"],
        )
        # Add standard
        self.node_std = KGNode(
            id="std_din_178",
            label="DIN EN ISO 178",
            entity_type=EntityType.STANDARD,
        )
        # Add passage
        self.node_chunk = KGNode(
            id="chunk_crystal_001",
            label="rauvisio_crystal_tech_0097",
            entity_type=EntityType.PASSAGE,
            properties={"chunk_id": "rauvisio_crystal_tech_0097", "source_pdf": "crystal_guide.pdf"},
        )

        for n in [self.node_prod, self.node_tech, self.node_proc, self.node_std, self.node_chunk]:
            self.kg.add_node(n)

        # Add edges
        self.kg.add_edge(KGEdge(
            source_id=self.node_tech.id,
            target_id=self.node_prod.id,
            relation=RelationType.BELONGS_TO_PRODUCT,
        ))
        self.kg.add_edge(KGEdge(
            source_id=self.node_tech.id,
            target_id=self.node_proc.id,
            relation=RelationType.INVOLVES_PROCESS,
        ))
        self.kg.add_edge(KGEdge(
            source_id=self.node_prod.id,
            target_id=self.node_std.id,
            relation=RelationType.GOVERNED_BY_STANDARD,
        ))
        self.kg.add_edge(KGEdge(
            source_id=self.node_std.id,
            target_id=self.node_chunk.id,
            relation=RelationType.GROUNDED_IN,
        ))

    def test_find_node_by_label_and_alias(self):
        self.assertEqual(self.kg.find_node("RAUVISIO crystal").id, "prod_crystal")
        self.assertEqual(self.kg.find_node("crystal glass").id, "prod_crystal")
        self.assertIsNone(self.kg.find_node("unknown entity"))

    def test_out_edges_traversal(self):
        out_edges = self.kg.get_out_edges("tech_drilling")
        self.assertEqual(len(out_edges), 2)
        target_ids = {n.id for n, _ in out_edges}
        self.assertIn("prod_crystal", target_ids)
        self.assertIn("proc_drill", target_ids)

    def test_serialization(self):
        d = self.kg.to_dict()
        kg_restored = TerminologyKG.from_dict(d)
        self.assertEqual(kg_restored.node_count(), self.kg.node_count())
        self.assertEqual(kg_restored.edge_count(), self.kg.edge_count())


class TestTKGEndToEnd(unittest.TestCase):
    """Integration tests for built TKG, SubgraphExtractor, and ProvenanceTracker."""

    @classmethod
    def setUpClass(cls):
        cls.kg = build_terminology_kg()
        cls.extractor = SubgraphExtractor(cls.kg)
        cls.provenance = ProvenanceTracker(cls.kg)

    def test_graph_counts(self):
        summary = self.kg.summary()
        self.assertGreater(summary["total_nodes"], 1000)
        self.assertGreater(summary["total_edges"], 1000)
        self.assertIn("product_family", summary["entity_types"])
        self.assertIn("standard", summary["entity_types"])
        self.assertIn("GOVERNED_BY_STANDARD", summary["relation_types"])

    def test_subgraph_extraction(self):
        query = "How does this metal-look material burn?"
        sub = self.extractor.extract_subgraph(query)
        self.assertIsInstance(sub, SubgraphResult)
        self.assertIn("RAUVISIO ferro", sub.product_families)
        self.assertIn("Fire Behavior & Safety", sub.processes)
        self.assertIn("DIN 4102", sub.expansion_query)

    def test_provenance_tracking(self):
        std_nodes = self.kg.get_nodes_by_type(EntityType.STANDARD)
        self.assertTrue(len(std_nodes) > 0)
        std_with_prov = [s for s in std_nodes if len(self.provenance.get_provenance_for_node(s.id)) > 0]
        self.assertTrue(len(std_with_prov) > 0)


if __name__ == "__main__":
    unittest.main()
