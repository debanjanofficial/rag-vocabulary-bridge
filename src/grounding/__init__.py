"""Grounding, claim attribution, and sentence-level NLI verification package.

Exports tools to decompose answers into atomic claims, evaluate factual
entailment with cross-encoder NLI models, bind citations to TKG provenance,
and calculate journal-grade faithfulness and hallucination metrics.
"""

from src.grounding.claims import ClaimStatement, extract_claims
from src.grounding.evaluator import GroundingEvaluator, GroundingReport
from src.grounding.nli import ClaimVerification, NLIVerifier

__all__ = [
    "ClaimStatement",
    "extract_claims",
    "ClaimVerification",
    "NLIVerifier",
    "GroundingEvaluator",
    "GroundingReport",
]
