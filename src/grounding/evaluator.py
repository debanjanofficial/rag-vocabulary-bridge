"""Comprehensive answer grounding evaluation and citation scoring module.

Calculates journal-grade metrics including Faithfulness (Grounding Ratio),
Hallucination Rate, Citation Precision/Recall, and renders claim-level
annotated visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.grounding.claims import ClaimStatement, extract_claims
from src.grounding.nli import ClaimVerification, NLIVerifier


@dataclass
class GroundingReport:
    """Consolidated grounding evaluation report for a generated answer."""

    claims: list[ClaimVerification]
    num_claims: int
    num_factual_claims: int
    num_entailed: int
    num_contradicted: int
    num_neutral: int
    faithfulness_ratio: float
    hallucination_ratio: float
    citation_precision: float
    citation_recall: float
    tkg_provenance_ratio: float
    annotated_markdown: str
    verified_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "num_claims": self.num_claims,
            "num_factual_claims": self.num_factual_claims,
            "num_entailed": self.num_entailed,
            "num_contradicted": self.num_contradicted,
            "num_neutral": self.num_neutral,
            "faithfulness_ratio": round(self.faithfulness_ratio, 4),
            "hallucination_ratio": round(self.hallucination_ratio, 4),
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "tkg_provenance_ratio": round(self.tkg_provenance_ratio, 4),
            "verified_citations": self.verified_citations,
            "claims": [c.to_dict() for c in self.claims],
            "annotated_markdown": self.annotated_markdown,
        }


class GroundingEvaluator:
    """Evaluates answer grounding, claim entailment, and citation accuracy."""

    def __init__(self, verifier: NLIVerifier | None = None) -> None:
        """Initialize evaluator with optional pre-configured NLI verifier."""
        self.verifier = verifier or NLIVerifier()

    def evaluate_answer(
        self,
        answer_text: str,
        docs: Sequence[dict[str, Any]],
        cited_chunk_ids: Sequence[str] | None = None,
        tkg_tracker: Any = None,
    ) -> GroundingReport:
        """Evaluate factual claims and citations in a generated answer.

        Args:
            answer_text: The generated text to evaluate.
            docs: The candidate manual passages used for grounding.
            cited_chunk_ids: Optional list of chunk IDs explicitly cited by LLM.
            tkg_tracker: Optional TKG ProvenanceTracker.

        Returns:
            A GroundingReport with claim-level verifications and metrics.
        """
        raw_claims = extract_claims(answer_text)

        if not raw_claims:
            return GroundingReport(
                claims=[],
                num_claims=0,
                num_factual_claims=0,
                num_entailed=0,
                num_contradicted=0,
                num_neutral=0,
                faithfulness_ratio=1.0,
                hallucination_ratio=0.0,
                citation_precision=1.0,
                citation_recall=1.0,
                tkg_provenance_ratio=0.0,
                annotated_markdown=answer_text,
                verified_citations=[],
            )

        verifications: list[ClaimVerification] = []
        verified_chunks: list[str] = []
        tkg_verified_count = 0

        for claim in raw_claims:
            ver = self.verifier.verify_claim(
                claim=claim,
                candidate_passages=docs,
                tkg_tracker=tkg_tracker,
            )
            verifications.append(ver)

            if ver.verdict == "entailment" and ver.supporting_chunk_id:
                if ver.supporting_chunk_id not in verified_chunks:
                    verified_chunks.append(ver.supporting_chunk_id)
                if ver.tkg_provenance_verified:
                    tkg_verified_count += 1

        num_claims = len(verifications)
        factual_claims = [c for c in verifications if c.verdict != "neutral" or c.confidence > 0.3]
        num_factual = len(factual_claims) or num_claims

        entailed = [c for c in verifications if c.verdict == "entailment"]
        contradicted = [c for c in verifications if c.verdict == "contradiction"]
        neutral = [c for c in verifications if c.verdict == "neutral"]

        num_ent = len(entailed)
        num_contra = len(contradicted)
        num_neut = len(neutral)

        faithfulness = (num_ent / num_factual) if num_factual > 0 else 1.0
        hallucination = (num_contra / num_factual) if num_factual > 0 else 0.0

        # Citation metrics
        llm_cites = set(cited_chunk_ids) if cited_chunk_ids else set()
        ver_cites = set(verified_chunks)

        if llm_cites:
            precision = len(llm_cites & ver_cites) / len(llm_cites)
            recall = (
                (len(llm_cites & ver_cites) / len(ver_cites))
                if ver_cites
                else 1.0
            )
        else:
            # If model didn't output explicit citations, evaluate coverage
            precision = 1.0 if num_ent > 0 else 0.0
            recall = 1.0 if num_ent == num_factual else (num_ent / max(1, num_factual))

        tkg_ratio = (tkg_verified_count / max(1, num_ent)) if num_ent > 0 else 0.0

        # Build annotated markdown
        annotated_md = self._render_annotated_markdown(verifications)

        return GroundingReport(
            claims=verifications,
            num_claims=num_claims,
            num_factual_claims=num_factual,
            num_entailed=num_ent,
            num_contradicted=num_contra,
            num_neutral=num_neut,
            faithfulness_ratio=faithfulness,
            hallucination_ratio=hallucination,
            citation_precision=precision,
            citation_recall=recall,
            tkg_provenance_ratio=tkg_ratio,
            annotated_markdown=annotated_md,
            verified_citations=verified_chunks,
        )

    def _render_annotated_markdown(
        self, verifications: list[ClaimVerification]
    ) -> str:
        """Render markdown text with colored attribution badges."""
        rendered_sentences: list[str] = []

        for v in verifications:
            sentence = v.text
            conf_pct = int(round(v.confidence * 100))

            if v.verdict == "entailment":
                chunk = v.supporting_chunk_id or f"doc_{v.supporting_doc_index}"
                badge = f" :green[**✓ [{chunk}]** ({conf_pct}% conf)]"
                if v.tkg_provenance_verified:
                    badge += " :violet[**[TKG Provenance]**]"
            elif v.verdict == "contradiction":
                badge = f" :red[**✗ [Contradicted]** ({conf_pct}% conf)]"
            else:
                badge = f" :orange[**? [Ungrounded]** ({conf_pct}% neutral)]"

            rendered_sentences.append(f"{sentence}{badge}")

        return " ".join(rendered_sentences)
