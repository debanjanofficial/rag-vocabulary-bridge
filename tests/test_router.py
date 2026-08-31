"""Unit and integration tests for the Adaptive Router module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.router.bm25_index import BM25Index, tokenize
from src.router.features import (
    FeatureVector,
    compute_expansion_drift_risk,
    compute_retrieval_entropy,
    extract_features,
)
from src.router.policy import (
    CalibratedPolicy,
    PolicyDecision,
    RoutingAction,
)
from src.router.router import AdaptiveRouter, RouterResult, route_query


class TestBM25Index(unittest.TestCase):
    """Tests for in-memory BM25 index and IDF computation."""

    def setUp(self):
        self.chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
        self.docs = [
            "REHAU Floating Shelves installation manual with wall studs.",
            "RAUVISIO crystal acrylic surface boards and cutting tools.",
            "Fire rating classification according to DIN 4102 standards.",
        ]
        self.index = BM25Index(self.chunk_ids, self.docs)

    def test_tokenize(self):
        tokens = tokenize("RAUVISIO-crystal, 19.8mm!")
        self.assertIn("rauvisio", tokens)
        self.assertIn("crystal", tokens)
        self.assertIn("19", tokens)
        self.assertIn("8mm", tokens)

    def test_search_exact_match(self):
        results = self.index.search("floating shelves wall studs", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0], "chunk_0")

    def test_query_specificity(self):
        spec = self.index.query_specificity("acrylic surface")
        self.assertGreaterEqual(spec, 0.0)
        self.assertLessEqual(spec, 1.0)


class TestFeatureExtraction(unittest.TestCase):
    """Tests for feature extraction and numerical bounds."""

    def test_entropy_sharp_vs_flat(self):
        sharp_hits = [
            {"score": 0.95, "chunk_id": "c1"},
            {"score": 0.20, "chunk_id": "c2"},
            {"score": 0.10, "chunk_id": "c3"},
        ]
        flat_hits = [
            {"score": 0.50, "chunk_id": "c1"},
            {"score": 0.50, "chunk_id": "c2"},
            {"score": 0.50, "chunk_id": "c3"},
        ]
        h_sharp = compute_retrieval_entropy(sharp_hits, top_k=3, temperature=0.1)
        h_flat = compute_retrieval_entropy(flat_hits, top_k=3, temperature=0.1)

        self.assertLess(h_sharp, h_flat)
        self.assertGreaterEqual(h_sharp, 0.0)
        self.assertLessEqual(h_flat, 1.0)

    def test_drift_risk(self):
        risk_high = compute_expansion_drift_risk("specialized tool", s_term=0.1)
        risk_low = compute_expansion_drift_risk("specialized tool", s_term=0.9)
        self.assertGreater(risk_high, risk_low)


class TestPolicyDecisions(unittest.TestCase):
    """Tests for calibrated policy boundary conditions."""

    def setUp(self):
        self.policy = CalibratedPolicy()

    def test_clarify_trigger(self):
        fv = FeatureVector(
            query="How to clean the surface?",
            s_term=0.2,
            c_prod=0.45,
            m_prod=0.03,  # Very narrow margin between top 2 products
            top_product="RAUVISIO crystal",
            j_agree=0.10,
            h_dense=0.85,  # High entropy
            r_drift=0.2,
        )
        decision = self.policy.decide(fv)
        self.assertEqual(decision.action, RoutingAction.CLARIFY)
        self.assertIsNotNone(decision.suggested_clarification)

    def test_passthrough_trigger(self):
        fv = FeatureVector(
            query="What is the reaction to fire classification of RAUVISIO noir?",
            s_term=0.1,
            c_prod=0.3,
            m_prod=0.1,
            top_product="RAUVISIO noir",
            j_agree=0.55,  # High agreement
            h_dense=0.3,
            r_drift=0.1,
        )
        decision = self.policy.decide(fv)
        self.assertEqual(decision.action, RoutingAction.PASSTHROUGH)

    def test_terminology_expansion_trigger(self):
        fv = FeatureVector(
            query="How do I stop the laminate from bending when I press it?",
            s_term=0.85,  # High colloquial term match
            c_prod=0.3,
            m_prod=0.1,
            top_product=None,
            j_agree=0.15,
            h_dense=0.4,
            r_drift=0.20,  # Low drift risk
            best_term_match={"preferred_term": "warpage"},
        )
        decision = self.policy.decide(fv)
        self.assertEqual(decision.action, RoutingAction.EXPAND_TERMINOLOGY)

    def test_metadata_filter_trigger(self):
        fv = FeatureVector(
            query="What colors can these handles come in?",
            s_term=0.1,
            c_prod=0.92,  # Strong product confidence
            m_prod=0.60,  # Clear margin
            top_product="Integrated Handles",
            j_agree=0.20,
            h_dense=0.3,
            r_drift=0.1,
        )
        decision = self.policy.decide(fv)
        self.assertEqual(decision.action, RoutingAction.FILTER_METADATA)
        self.assertEqual(decision.target_product, "Integrated Handles")


class TestAdaptiveRouterEndToEnd(unittest.TestCase):
    """End-to-end routing integration tests."""

    def test_empty_query(self):
        res = route_query("")
        self.assertEqual(res.action, RoutingAction.PASSTHROUGH)
        self.assertEqual(res.final_query, "")

    def test_real_colloquial_query_routing(self):
        query = "How do I make sure the edges look good when cutting these panels?"
        res = route_query(query, llm_available=False)
        self.assertIsInstance(res, RouterResult)
        self.assertIn(res.action, list(RoutingAction))
        self.assertGreaterEqual(len(res.retrieval_queries), 1)

        d = res.to_dict()
        self.assertIn("action", d)
        self.assertIn("features", d)
        self.assertIn("s_term", d["features"])


if __name__ == "__main__":
    unittest.main()
