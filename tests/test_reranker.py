"""Unit tests for the Path-Aware Neural Reranker module."""

import unittest

from src.reranker import rerank_documents
from src.tkg.schema import SubgraphResult


class TestPathAwareReranker(unittest.TestCase):
    """Tests for cross-encoder reranking with multi-hop path injection."""

    def setUp(self):
        self.sample_docs = [
            {
                "chunk_id": "chunk_cutting_001",
                "document": (
                    "To achieve clean chip-free cuts on RAUVISIO ingrain "
                    "wood fiber laminate, use a carbide-tipped saw blade "
                    "with a sacrificial underlay panel."
                ),
            },
            {
                "chunk_id": "chunk_cleaning_002",
                "document": (
                    "Daily cleaning of surfaces requires a soft microfiber "
                    "cloth and mild soapy water."
                ),
            },
            {
                "chunk_id": "chunk_standard_003",
                "document": (
                    "Reaction to fire classification according to DIN 4102-1 "
                    "is rated as class B2 standard flammability."
                ),
            },
        ]
        self.query = (
            "How do I make sure the edges look good when cutting my "
            "wood-grain panels?"
        )

    def test_empty_documents(self):
        res = rerank_documents(self.query, [])
        self.assertEqual(res, [])

    def test_baseline_rerank(self):
        res = rerank_documents(self.query, self.sample_docs, top_n=2)
        self.assertEqual(len(res), 2)
        self.assertIn("rerank_score", res[0])
        self.assertGreaterEqual(res[0]["rerank_score"], res[1]["rerank_score"])
        self.assertFalse(res[0].get("graph_provenance", False))

    def test_path_injected_rerank(self):
        path_str = (
            "wood-look panels -> RAUVISIO ingrain -> Cutting & Machining"
        )
        res = rerank_documents(
            self.query,
            self.sample_docs,
            top_n=3,
            path_context=path_str,
        )
        self.assertEqual(len(res), 3)
        self.assertTrue(all("path_context" in d for d in res))
        self.assertIn("wood-look panels", res[0]["path_context"])
        # Cutting document should rank first
        self.assertEqual(res[0]["chunk_id"], "chunk_cutting_001")

    def test_provenance_boost(self):
        target_chunk = "chunk_standard_003"
        res_no_boost = rerank_documents(
            self.query,
            self.sample_docs,
            top_n=3,
            provenance_chunk_ids=None,
        )
        score_before = next(
            d["rerank_score"] for d in res_no_boost
            if d["chunk_id"] == target_chunk
        )

        boost_val = 0.50
        res_boost = rerank_documents(
            self.query,
            self.sample_docs,
            top_n=3,
            provenance_chunk_ids=[target_chunk],
            provenance_boost=boost_val,
        )
        target_doc = next(
            d for d in res_boost if d["chunk_id"] == target_chunk
        )
        self.assertTrue(target_doc["graph_provenance"])
        self.assertAlmostEqual(
            target_doc["rerank_score"],
            score_before + boost_val,
            places=3,
        )

    def test_subgraph_result_interoperability(self):
        sub = SubgraphResult(
            matched_nodes=[],
            subgraph_nodes=[],
            subgraph_edges=[],
            preferred_terms=["RAUVISIO ingrain"],
            product_families=["RAUVISIO ingrain"],
            processes=["Cutting & Machining"],
            standards=["DIN 4102"],
            provenance_chunk_ids=["chunk_cutting_001"],
            expansion_query="sample expansion",
            linearized_paths=[
                "wood-look panels -> RAUVISIO ingrain -> Cutting & Machining"
            ],
        )
        res = rerank_documents(
            self.query,
            self.sample_docs,
            top_n=3,
            path_context=sub,
        )
        self.assertEqual(len(res), 3)
        # Verify grounded chunk in subgraph was marked
        cutting_doc = next(
            d for d in res if d["chunk_id"] == "chunk_cutting_001"
        )
        self.assertTrue(cutting_doc["graph_provenance"])
        self.assertIn("Path: wood-look panels", cutting_doc["path_context"])


if __name__ == "__main__":
    unittest.main()
