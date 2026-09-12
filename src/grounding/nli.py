"""Natural Language Inference (NLI) claim verifier and attribution engine.

Uses cross-encoder NLI models to assess entailment, neutral, and contradiction
probabilities between retrieved manual passages and generated claim statements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from config import (
    MAX_CANDIDATE_PASSAGES_FOR_NLI,
    NLI_CONTRADICTION_THRESHOLD,
    NLI_ENTAILMENT_THRESHOLD,
    NLI_MODEL,
)
from src.grounding.claims import ClaimStatement

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerification:
    """NLI verification result for an individual claim."""

    claim_id: int
    text: str
    original_text: str
    verdict: str  # "entailment", "neutral", "contradiction"
    confidence: float
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    supporting_chunk_id: str | None = None
    supporting_doc_index: int | None = None
    supporting_excerpt: str | None = None
    tkg_provenance_verified: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "original_text": self.original_text,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "entailment_prob": round(self.entailment_prob, 4),
            "neutral_prob": round(self.neutral_prob, 4),
            "contradiction_prob": round(self.contradiction_prob, 4),
            "supporting_chunk_id": self.supporting_chunk_id,
            "supporting_doc_index": self.supporting_doc_index,
            "supporting_excerpt": self.supporting_excerpt,
            "tkg_provenance_verified": self.tkg_provenance_verified,
            "details": self.details,
        }


class NLIVerifier:
    """Evaluates factual entailment between text passages and generated claims."""

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        entailment_threshold: float = NLI_ENTAILMENT_THRESHOLD,
        contradiction_threshold: float = NLI_CONTRADICTION_THRESHOLD,
        max_passages: int = MAX_CANDIDATE_PASSAGES_FOR_NLI,
    ) -> None:
        """Initialize verifier with model specifications and thresholds."""
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.max_passages = max_passages
        self._model = None
        self._label_map: dict[str, int] = {}

    def _load_model(self) -> Any:
        """Load cross-encoder model on demand."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading NLI CrossEncoder model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)

            # Map config id2label to standard labels
            id2label = getattr(self._model.config, "id2label", None)
            if id2label:
                self._label_map = {
                    name.lower(): idx for idx, name in id2label.items()
                }
            else:
                # Default mapping for DeBERTa / RoBERTa NLI cross encoders
                self._label_map = {
                    "contradiction": 0,
                    "entailment": 1,
                    "neutral": 2,
                }
        return self._model

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply numerically stable softmax along the last axis."""
        shift = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shift)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def predict_pairs(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[dict[str, float]]:
        """Predict probabilities for a batch of (premise, hypothesis) pairs."""
        if not pairs:
            return []

        model = self._load_model()
        raw_scores = model.predict(list(pairs), show_progress_bar=False)
        logits = np.array(raw_scores)
        probs = self._softmax(logits)

        c_idx = self._label_map.get("contradiction", 0)
        e_idx = self._label_map.get("entailment", 1)
        n_idx = self._label_map.get("neutral", 2)

        results: list[dict[str, float]] = []
        for row in probs:
            results.append(
                {
                    "contradiction": float(row[c_idx]),
                    "entailment": float(row[e_idx]),
                    "neutral": float(row[n_idx]),
                }
            )
        return results

    def verify_claim(
        self,
        claim: ClaimStatement,
        candidate_passages: Sequence[dict[str, Any]],
        tkg_tracker: Any = None,
    ) -> ClaimVerification:
        """Verify an individual claim against candidate passages.

        Args:
            claim: Extracted atomic claim statement.
            candidate_passages: Top retrieved passages with chunk metadata.
            tkg_tracker: Optional ProvenanceTracker from Pillar 3.

        Returns:
            ClaimVerification with verdict, confidence, and citation binding.
        """
        if not candidate_passages:
            return ClaimVerification(
                claim_id=claim.claim_id,
                text=claim.text,
                original_text=claim.original_text,
                verdict="neutral",
                confidence=1.0,
                entailment_prob=0.0,
                neutral_prob=1.0,
                contradiction_prob=0.0,
            )

        passages = list(candidate_passages[: self.max_passages])
        # Form (premise, hypothesis) pairs with windowing for long passages
        pairs: list[tuple[str, str]] = []
        pair_metadata: list[tuple[int, str]] = []  # (doc_idx, excerpt)

        for p_idx, p in enumerate(passages):
            doc_text = (p.get("document") or p.get("text") or "").strip()
            windows = self._extract_windows(doc_text, claim.text)
            for w in windows:
                pairs.append((w, claim.text))
                pair_metadata.append((p_idx, w))

        evaluations = self.predict_pairs(pairs)

        best_entail_doc_idx = -1
        max_entail_prob = -1.0
        best_entail_excerpt = ""
        best_entail_scores = {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}

        best_contra_doc_idx = -1
        max_contra_prob = -1.0
        best_contra_excerpt = ""
        best_contra_scores = {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}

        # Extract content words from claim to check topical relevance
        import re
        claim_content_words = {
            w.lower()
            for w in re.findall(r"\w+", claim.text)
            if len(w) > 3
            and w.lower()
            not in {"this", "that", "with", "from", "your", "they", "will", "have"}
        }

        for (p_idx, window_text), scores in zip(pair_metadata, evaluations):
            e_prob = scores["entailment"]
            c_prob = scores["contradiction"]
            if e_prob > max_entail_prob:
                max_entail_prob = e_prob
                best_entail_doc_idx = p_idx
                best_entail_excerpt = window_text
                best_entail_scores = scores

            window_content_words = {
                w.lower()
                for w in re.findall(r"\w+", window_text)
                if len(w) > 3
            }
            # Only consider contradiction if the window is top-ranked or topical
            is_topical = (p_idx == 0) or bool(claim_content_words & window_content_words)
            if is_topical and c_prob > max_contra_prob:
                max_contra_prob = c_prob
                best_contra_doc_idx = p_idx
                best_contra_excerpt = window_text
                best_contra_scores = scores

        # Determine verdict: entailment takes precedence if any candidate supports the claim
        if max_entail_prob >= self.entailment_threshold:
            verdict = "entailment"
            confidence = max_entail_prob
            ref_idx = best_entail_doc_idx
            scores = best_entail_scores
            excerpt = best_entail_excerpt[:240]
        elif (
            max_contra_prob >= self.contradiction_threshold
            and max_contra_prob > max_entail_prob
            and max_contra_prob > best_contra_scores.get("neutral", 0.0)
        ):
            verdict = "contradiction"
            confidence = max_contra_prob
            ref_idx = best_contra_doc_idx
            scores = best_contra_scores
            excerpt = best_contra_excerpt[:240]
        else:
            verdict = "neutral"
            ref_idx = best_entail_doc_idx if best_entail_doc_idx >= 0 else 0
            scores = best_entail_scores if best_entail_doc_idx >= 0 else best_contra_scores
            confidence = scores["neutral"]
            excerpt = best_entail_excerpt[:240] if best_entail_excerpt else ""

        target_doc = passages[ref_idx]
        chunk_id = target_doc.get("chunk_id")
        full_doc = target_doc.get("document") or target_doc.get("text") or ""
        excerpt = full_doc.strip()[:240]
        if len(full_doc) > 240:
            excerpt += "…"

        # Check TKG provenance alignment if tracker is provided
        tkg_aligned = False
        if tkg_tracker and chunk_id:
            try:
                # Check if claim entities bind to this chunk
                tkg_aligned = self._check_tkg_alignment(
                    claim.text, chunk_id, tkg_tracker
                )
            except Exception as e:
                logger.debug("TKG alignment check skipped: %s", e)

        return ClaimVerification(
            claim_id=claim.claim_id,
            text=claim.text,
            original_text=claim.original_text,
            verdict=verdict,
            confidence=confidence,
            entailment_prob=scores["entailment"],
            neutral_prob=scores["neutral"],
            contradiction_prob=scores["contradiction"],
            supporting_chunk_id=chunk_id if verdict == "entailment" else None,
            supporting_doc_index=(ref_idx + 1)
            if verdict == "entailment"
            else None,
            supporting_excerpt=excerpt if verdict == "entailment" else None,
            tkg_provenance_verified=tkg_aligned,
            details={
                "evaluated_passages": len(passages),
                "all_scores": evaluations,
            },
        )

    def _check_tkg_alignment(
        self, claim_text: str, chunk_id: str, tkg_tracker: Any
    ) -> bool:
        """Verify whether chunk_id is linked to entities present in the claim."""
        if not hasattr(tkg_tracker, "kg"):
            return False

        from src.tkg.schema import EntityType

        low_claim = claim_text.lower()
        matched_entities = []
        for node in tkg_tracker.kg.graph.nodes.values():
            label = node.get("label", "")
            if len(label) >= 4 and label.lower() in low_claim:
                matched_entities.append(node.get("id"))

        if not matched_entities:
            return False

        for node_id in matched_entities:
            provenance = tkg_tracker.get_provenance_for_node(node_id)
            for p in provenance:
                if p.get("chunk_id") == chunk_id:
                    return True
        return False

    def _extract_windows(self, doc_text: str, claim_text: str) -> list[str]:
        """Extract focused candidate premise windows from a passage."""
        import re

        clean_doc = doc_text.strip()
        if not clean_doc:
            return []
        if len(clean_doc) <= 600:
            return [clean_doc]

        # Split into sentence, bullet point, or paragraph segments
        raw_segments = [
            s.strip()
            for s in re.split(r"(?<=[.!?\n▪•])\s+|(?=[▪•])", clean_doc)
            if len(s.strip()) > 15
        ]

        if not raw_segments:
            return [clean_doc[:800]]

        # Score segments by lexical overlap with claim
        claim_words = set(re.findall(r"\w+", claim_text.lower()))
        scored: list[tuple[int, str]] = []
        for s in raw_segments:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(claim_words & s_words)
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Select top 3 most overlapping segments + beginning of document
        windows: list[str] = []
        for _, s in scored[:3]:
            if s not in windows:
                windows.append(s)

        doc_prefix = clean_doc[:600]
        if doc_prefix not in windows:
            windows.append(doc_prefix)

        return windows
