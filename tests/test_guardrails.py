"""Unit and integration tests for Pillar 6 Deterministic Safety Guardrails.

Verifies zero-tolerance policy enforcement against prohibited chemicals,
negation context analysis, and deterministic factory safety overrides.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from src.generator import generate_answer
from src.tkg.feedback import FeedbackManager
from src.tkg.graph import TerminologyKG
from src.tkg.guardrails import SafetyGuardrail, SafetyReport


class TestSafetyGuardrail(unittest.TestCase):
    """Test suite for safety guardrails and deterministic rule enforcement."""

    def setUp(self) -> None:
        """Create temporary feedback manager with verified safety rules."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rules_path = os.path.join(
            self.temp_dir.name, "expert_corrections.json"
        )
        self.graph_path = os.path.join(
            self.temp_dir.name, "terminology_graph.json"
        )

        self.kg = TerminologyKG()
        self.kg.save_json(self.graph_path)

        self.fm = FeedbackManager(
            rules_path=self.rules_path,
            graph_path=self.graph_path,
        )

        # Inject sample factory safety rule
        self.fm.submit_correction(
            query="Can I clean shade panels with window cleaner?",
            product="RAUVISIO shade",
            contradicted_claim="Window cleaner is acceptable when diluted.",
            approved_rule=(
                "Never use window cleaner diluted with water on RAUVISIO "
                "shade surfaces. Use only mild soap and a damp microfiber "
                "cloth."
            ),
            author="Plant Safety Director",
            source_ref="RAUVISIO Shade Cleaning Spec 2025",
        )

        self.guardrail = SafetyGuardrail(feedback_manager=self.fm)

    def tearDown(self) -> None:
        """Clean up temporary test environment."""
        self.temp_dir.cleanup()

    def test_prohibited_terms_extraction(self) -> None:
        """Verify extraction of prohibited chemical terms from active rules."""
        rules = self.fm.load_rules()
        self.assertEqual(len(rules), 1)

        prohibited = self.guardrail.extract_prohibited_terms(rules[0])
        self.assertIn("window cleaner", prohibited)

    def test_negation_detection_safe_affirmations(self) -> None:
        """Verify cautionary warnings are correctly classified as safe."""
        safe_sentence_prefix = (
            "Never use window cleaner on high-gloss shade surfaces."
        )
        self.assertTrue(
            self.guardrail.is_negated_or_safe(
                safe_sentence_prefix, "window cleaner"
            )
        )

        safe_sentence_postfix = (
            "Application of window cleaner is forbidden on these panels."
        )
        self.assertTrue(
            self.guardrail.is_negated_or_safe(
                safe_sentence_postfix, "window cleaner"
            )
        )

    def test_negation_detection_unsafe_affirmation(self) -> None:
        """Verify hazardous affirmative recommendations trigger violation."""
        unsafe_sentence = (
            "You can safely use window cleaner to wipe away greasy stains."
        )
        self.assertFalse(
            self.guardrail.is_negated_or_safe(
                unsafe_sentence, "window cleaner"
            )
        )

    def test_validate_safe_answer(self) -> None:
        """Verify compliant answer passes through without interception."""
        safe_answer = (
            "For RAUVISIO shade panels, never use window cleaner. Clean "
            "with a damp microfiber cloth and mild soap."
        )
        report: SafetyReport = self.guardrail.validate_answer(
            answer=safe_answer,
            query="Can I clean shade panels with window cleaner?",
            product="RAUVISIO shade",
            auto_sanitize=True,
        )

        self.assertTrue(report.is_safe)
        self.assertFalse(report.intercepted)
        self.assertEqual(len(report.violations), 0)
        self.assertEqual(report.compliance_score, 1.0)
        self.assertEqual(report.sanitized_answer, safe_answer)

    def test_validate_unsafe_answer_triggers_override(self) -> None:
        """Verify unsafe answer is deterministically overridden."""
        unsafe_answer = (
            "You should wipe the surface with window cleaner every week. "
            "Afterwards, inspect the finish for clarity."
        )
        report: SafetyReport = self.guardrail.validate_answer(
            answer=unsafe_answer,
            query="Can I clean shade panels with window cleaner?",
            product="RAUVISIO shade",
            auto_sanitize=True,
        )

        self.assertFalse(report.is_safe)
        self.assertTrue(report.intercepted)
        self.assertEqual(len(report.violations), 1)
        self.assertEqual(
            report.violations[0].prohibited_term, "window cleaner"
        )
        self.assertIn(
            "[MANDATORY FACTORY SAFETY OVERRIDE]",
            report.sanitized_answer,
        )
        self.assertNotIn(
            "You should wipe the surface with window cleaner",
            report.sanitized_answer,
        )
        self.assertIn(
            "Afterwards, inspect the finish for clarity.",
            report.sanitized_answer,
        )

    def test_unrelated_query_passes_unconstrained(self) -> None:
        """Verify queries without matching negative constraints are safe."""
        harmless_answer = "RAUVISIO shade is available in 12 matte finishes."
        report = self.guardrail.validate_answer(
            answer=harmless_answer,
            query="What colors are available?",
            product="RAUVISIO shade",
            auto_sanitize=True,
        )
        self.assertTrue(report.is_safe)
        self.assertFalse(report.intercepted)
        self.assertEqual(report.sanitized_answer, harmless_answer)

    def test_generate_answer_safety_integration(self) -> None:
        """Verify end-to-end integration with answer generation pipeline."""
        from unittest.mock import patch

        sample_docs = [
            {
                "document": "Factory maintenance manual for RAUVISIO shade.",
                "product": "RAUVISIO shade",
                "chunk_id": "shade_clean_001",
            }
        ]

        with patch("src.generator.chat_text") as mock_chat:
            mock_chat.return_value = (
                "You can safely use window cleaner to wipe the surface."
            )
            result = generate_answer(
                query="Can I clean shade panels with window cleaner?",
                docs=sample_docs,
                llm_available=True,
                verify_grounding=False,
            )

        self.assertIn("safety", result)
        safety_data = result["safety"]
        self.assertIsNotNone(safety_data)
        self.assertTrue(safety_data["intercepted"])
        self.assertFalse(safety_data["is_safe"])
        self.assertIn(
            "[MANDATORY FACTORY SAFETY OVERRIDE]",
            result["answer"],
        )


if __name__ == "__main__":
    unittest.main()
