"""Unit and integration tests for Human-in-the-Loop Feedback and Knowledge Graph Rule Injection."""

import os
import tempfile
import unittest

from src.tkg.feedback import FeedbackManager
from src.tkg.graph import TerminologyKG
from src.tkg.schema import EntityType, RelationType


class TestFeedbackManager(unittest.TestCase):
    """Test suite for human feedback, rule store, and TKG constraint injection."""

    def setUp(self):
        """Create temporary rule store and test knowledge graph."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rules_path = os.path.join(self.temp_dir.name, "expert_corrections.json")
        self.graph_path = os.path.join(self.temp_dir.name, "terminology_graph.json")

        # Initialize a small test knowledge graph
        self.kg = TerminologyKG()
        self.kg.save_json(self.graph_path)

        self.fm = FeedbackManager(
            rules_path=self.rules_path,
            graph_path=self.graph_path,
        )

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_submit_correction_updates_rules_and_graph(self):
        """Test submitting correction saves rule and injects constraint into TKG."""
        rule = self.fm.submit_correction(
            query="Can I clean shade panels with window cleaner?",
            product="RAUVISIO shade",
            contradicted_claim="Window cleaner diluted with water is acceptable.",
            approved_rule="Do not use window cleaner diluted with water on RAUVISIO shade; clean only with mild soap and damp microfiber.",
            author="Quality Inspector Mike",
            source_ref="RAUVISIO Cleaning Instructions Page 1",
        )

        self.assertIn("id", rule)
        self.assertEqual(rule["product"], "RAUVISIO shade")
        self.assertTrue(rule["active"])

        # Verify rule exists in persistent file
        loaded_rules = self.fm.load_rules()
        self.assertEqual(len(loaded_rules), 1)
        self.assertEqual(loaded_rules[0]["author"], "Quality Inspector Mike")

        # Verify Knowledge Graph was updated with constraint node and edge
        updated_kg = TerminologyKG.load_json(self.graph_path)
        constraint_nodes = updated_kg.get_nodes_by_type(EntityType.CONSTRAINT)
        self.assertEqual(len(constraint_nodes), 1)
        self.assertIn("Do not use window cleaner", constraint_nodes[0].definition)

        # Check for outgoing FORBIDS_AGENT edge from product
        prod_node = updated_kg.find_node("RAUVISIO shade")
        self.assertIsNotNone(prod_node)
        out_edges = updated_kg.get_out_edges(
            prod_node.id, relation=RelationType.FORBIDS_AGENT
        )
        self.assertEqual(len(out_edges), 1)
        self.assertEqual(out_edges[0][0].id, constraint_nodes[0].id)

    def test_find_matching_rules(self):
        """Test finding rules based on query and product overlap."""
        self.fm.submit_correction(
            query="Can I clean shade panels with window cleaner?",
            product="RAUVISIO shade",
            contradicted_claim="Window cleaner is fine",
            approved_rule="Never use window cleaner on RAUVISIO shade.",
        )

        # Test exact match
        matches = self.fm.find_matching_rules(
            "Can I clean shade panels with window cleaner?",
            product="RAUVISIO shade",
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(
            matches[0]["approved_rule"],
            "Never use window cleaner on RAUVISIO shade.",
        )

        # Test partial keyword match
        matches_partial = self.fm.find_matching_rules(
            "clean shade with window cleaner",
            product="RAUVISIO shade",
        )
        self.assertEqual(len(matches_partial), 1)

        # Test unrelated query returns no match
        matches_unrelated = self.fm.find_matching_rules(
            "How do I cut crystal glass?",
            product="RAUVISIO crystal",
        )
        self.assertEqual(len(matches_unrelated), 0)


if __name__ == "__main__":
    unittest.main()
