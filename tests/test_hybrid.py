"""Unit and integration tests for Hybrid Retrieval with Reciprocal Rank Fusion."""

import unittest

from src.retriever import retrieve_full_hybrid, retrieve_hybrid, retrieve_routed
from src.router.router import RouterResult, RoutingAction
from src.router.policy import PolicyDecision
from src.router.features import FeatureVector


class TestHybridRetrieval(unittest.TestCase):
    """Test suite for BM25 + Dense Reciprocal Rank Fusion retrieval."""

    def test_retrieve_hybrid_returns_top_k(self):
        query = "How do I cut crystal glass panels?"
        cids = retrieve_hybrid(query, top_k=5, alpha=0.5)
        self.assertEqual(len(cids), 5)
        self.assertTrue(all(isinstance(c, str) and len(c) > 0 for c in cids))

    def test_retrieve_full_hybrid_structure(self):
        query = "How do I cut crystal glass panels?"
        docs = retrieve_full_hybrid(query, top_k=5, alpha=0.5)
        self.assertEqual(len(docs), 5)
        for d in docs:
            self.assertIn("chunk_id", d)
            self.assertIn("rrf_score", d)
            self.assertIn("channel_origin", d)
            self.assertIn(d["channel_origin"], ("both", "dense_only", "bm25_only"))
            self.assertGreater(d["rrf_score"], 0.0)

    def test_alpha_weighting_extreme_values(self):
        query = "acclimation room temperature 48 hours"
        # alpha = 1.0 is dense only weighting
        dense_docs = retrieve_full_hybrid(query, top_k=5, alpha=1.0)
        # alpha = 0.0 is bm25 only weighting
        bm25_docs = retrieve_full_hybrid(query, top_k=5, alpha=0.0)

        self.assertEqual(len(dense_docs), 5)
        self.assertEqual(len(bm25_docs), 5)
        self.assertGreater(dense_docs[0]["rrf_score"], 0.0)
        self.assertGreater(bm25_docs[0]["rrf_score"], 0.0)

    def test_decoupled_lexical_dense_queries(self):
        colloquial_q = "How does this metal-look material burn?"
        lexical_q = "How does this metal-look material burn? (RAUVISIO ferro DIN 4102-B2 ASTM E84)"

        docs = retrieve_full_hybrid(
            query=colloquial_q,
            lexical_query=lexical_q,
            top_k=5,
            alpha=0.5,
        )
        self.assertEqual(len(docs), 5)
        # Verify that ferro / fire safety chunk is prioritized
        retrieved_ids = [d["chunk_id"] for d in docs]
        self.assertTrue(any("ferro" in cid for cid in retrieved_ids))

    def test_metadata_filtering(self):
        query = "What are the allowed tolerances?"
        docs = retrieve_full_hybrid(
            query=query,
            product="RAUVISIO crystal",
            top_k=5,
        )
        self.assertTrue(len(docs) > 0)
        for d in docs:
            self.assertEqual(d.get("product"), "RAUVISIO crystal")

    def test_routed_hybrid_integration(self):
        dummy_decision = PolicyDecision(
            action=RoutingAction.EXPAND_TERMINOLOGY,
            confidence=0.85,
            reason="Test rationale",
        )

        dummy_features = FeatureVector(
            query="How does metal-look board burn?",
            s_term=0.7,
            c_prod=0.9,
            m_prod=0.8,
            top_product="RAUVISIO ferro",
            j_agree=0.2,
            h_dense=0.3,
            r_drift=0.1,
        )

        routed_res = RouterResult(
            original_query="How does metal-look board burn?",
            final_query="How does metal-look board burn?",
            action=RoutingAction.EXPAND_TERMINOLOGY,
            decision=dummy_decision,
            features=dummy_features,
            retrieval_queries=["How does metal-look board burn?", "DIN 4102"],
            lexical_query="How does metal-look board burn? (RAUVISIO ferro DIN 4102)",
            is_hybrid=True,
        )
        cids = retrieve_routed(routed_res, top_k=5)
        self.assertEqual(len(cids), 5)


if __name__ == "__main__":
    unittest.main()
