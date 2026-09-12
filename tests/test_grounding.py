"""Unit and integration tests for Answer Grounding & Sentence-Level NLI Citations."""

import unittest

from src.grounding.claims import ClaimStatement, extract_claims
from src.grounding.evaluator import GroundingEvaluator
from src.grounding.nli import NLIVerifier


class TestGroundingModule(unittest.TestCase):
    """Test suite for claim extraction, NLI verifier, and grounding metrics."""

    @classmethod
    def setUpClass(cls):
        """Initialize NLI verifier once for test suite."""
        cls.verifier = NLIVerifier()
        cls.evaluator = GroundingEvaluator(verifier=cls.verifier)

    def test_claim_extraction_preserves_technical_standards(self):
        """Test claim segmentation preserves abbreviations, decimals, and standards."""
        raw_text = (
            "Here is the technical specification:\n"
            "- RAUVISIO crystal complies with DIN 4102-B2 flammability standard.\n"
            "- The material thickness is approx. 2.5 mm with +/- 0.2 mm tolerance.\n"
            "- Saw blades should be carbide-tipped (e.g. triple chip teeth).\n"
            "Sources: [1], [2]\n"
        )
        claims = extract_claims(raw_text)

        self.assertEqual(len(claims), 3)
        self.assertIn("DIN 4102-B2", claims[0].text)
        self.assertIn("approx. 2.5 mm", claims[1].text)
        self.assertIn("e.g. triple chip teeth", claims[2].text)
        for c in claims:
            self.assertTrue(c.is_factual)
            self.assertNotIn("Sources:", c.text)

    def test_claim_extraction_empty_or_scaffolding(self):
        """Test extraction ignores purely conversational scaffolding and empty lines."""
        text = "Sources: [1], [2]\nAccording to the manual:\n\n"
        claims = extract_claims(text)
        self.assertEqual(len(claims), 0)

    def test_nli_entailment_verification(self):
        """Test NLI verifier correctly identifies entailed factual claims."""
        passages = [
            {
                "chunk_id": "chunk_sawing_01",
                "document": (
                    "When machining RAUVISIO crystal, use carbide-tipped "
                    "circular saw blades with alternating or triple-chip teeth."
                ),
            },
            {
                "chunk_id": "chunk_cleaning_02",
                "document": "Clean the surface with warm water and a soft cloth.",
            },
        ]
        claim = ClaimStatement(
            claim_id=1,
            text="Carbide-tipped saw blades should be used for cutting RAUVISIO crystal.",
            original_text="Carbide-tipped saw blades should be used for cutting RAUVISIO crystal.",
            is_factual=True,
        )

        verification = self.verifier.verify_claim(claim, passages)

        self.assertEqual(verification.verdict, "entailment")
        self.assertGreater(verification.entailment_prob, 0.70)
        self.assertEqual(verification.supporting_chunk_id, "chunk_sawing_01")
        self.assertEqual(verification.supporting_doc_index, 1)

    def test_nli_contradiction_detection(self):
        """Test NLI verifier detects direct contradictions and flags hallucinations."""
        passages = [
            {
                "chunk_id": "chunk_temp_01",
                "document": (
                    "Do not use abrasive cleaning agents, steel wool, or "
                    "solvents containing acetone on RAUVISIO crystal surfaces."
                ),
            }
        ]
        claim = ClaimStatement(
            claim_id=1,
            text="Acetone and abrasive steel wool are recommended to clean RAUVISIO crystal.",
            original_text="Acetone and abrasive steel wool are recommended to clean RAUVISIO crystal.",
            is_factual=True,
        )

        verification = self.verifier.verify_claim(claim, passages)

        self.assertEqual(verification.verdict, "contradiction")
        self.assertGreater(verification.contradiction_prob, 0.60)

    def test_nli_neutral_ungrounded_detection(self):
        """Test NLI verifier marks unsupported claims as neutral."""
        passages = [
            {
                "chunk_id": "chunk_edge_01",
                "document": (
                    "Edgebands can be applied using PUR adhesive or "
                    "laser edgebanding machinery."
                ),
            }
        ]
        claim = ClaimStatement(
            claim_id=1,
            text="The boards are manufactured in an offshore facility in Tokyo.",
            original_text="The boards are manufactured in an offshore facility in Tokyo.",
            is_factual=True,
        )

        verification = self.verifier.verify_claim(claim, passages)

        self.assertEqual(verification.verdict, "neutral")
        self.assertIsNone(verification.supporting_chunk_id)

    def test_grounding_evaluator_report(self):
        """Test overall report metrics: faithfulness, hallucination, and citations."""
        passages = [
            {
                "chunk_id": "chunk_crystal_cut",
                "document": (
                    "When machining RAUVISIO crystal, use carbide-tipped "
                    "circular saw blades."
                ),
            }
        ]
        answer = (
            "Carbide-tipped saw blades should be used for cutting RAUVISIO crystal. "
            "Also, Tokyo offshore facilities manufacture all parts."
        )

        report = self.evaluator.evaluate_answer(
            answer_text=answer,
            docs=passages,
            cited_chunk_ids=["chunk_crystal_cut"],
        )

        self.assertEqual(report.num_claims, 2)
        self.assertEqual(report.num_entailed, 1)
        self.assertEqual(report.num_neutral, 1)
        self.assertEqual(report.num_contradicted, 0)
        self.assertAlmostEqual(report.faithfulness_ratio, 0.5, places=2)
        self.assertAlmostEqual(report.hallucination_ratio, 0.0, places=2)
        self.assertIn("chunk_crystal_cut", report.verified_citations)
        self.assertIn(":green[**✓", report.annotated_markdown)
        self.assertIn(":orange[**?", report.annotated_markdown)

        # Ensure dictionary conversion is JSON serializable
        report_dict = report.to_dict()
        self.assertIn("faithfulness_ratio", report_dict)
        self.assertEqual(len(report_dict["claims"]), 2)


if __name__ == "__main__":
    unittest.main()
