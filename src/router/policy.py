"""Decision policy and action taxonomy for the Adaptive Router.

Implements the calibrated risk-controlled policy mapping FeatureVector
instances to one of five discrete execution actions:
  1) PASSTHROUGH: Zero query modification (preserves exact technical keywords).
  2) EXPAND_TERMINOLOGY: In-place parenthetical terminology graph expansion.
  3) FILTER_METADATA: Strict product metadata constraint with intent ranking.
  4) EXPAND_SEMANTIC: Intent-guided procedural query reformulation.
  5) CLARIFY: Active disambiguation question when product collision occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.router.features import FeatureVector


class RoutingAction(str, Enum):
    """Discrete routing actions available in the adaptive framework."""

    PASSTHROUGH = "passthrough"
    EXPAND_TERMINOLOGY = "expand_terminology"
    FILTER_METADATA = "filter_metadata"
    EXPAND_SEMANTIC = "expand_semantic"
    CLARIFY = "clarify"


@dataclass
class PolicyDecision:
    """Outcome of a routing policy decision with rationale and metadata."""

    action: RoutingAction
    confidence: float
    reason: str
    target_product: str | None = None
    suggested_clarification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to a JSON-serializable dictionary."""
        return {
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "target_product": self.target_product,
            "suggested_clarification": self.suggested_clarification,
        }


class CalibratedPolicy:
    """Calibrated rule-based policy with risk-bounding threshold parameters."""

    def __init__(
        self,
        tau_clarify_prod: float = 0.35,
        tau_clarify_margin: float = 0.10,
        tau_clarify_entropy: float = 0.65,
        tau_agree_passthrough: float = 0.30,
        tau_gap_passthrough: float = 0.40,
        tau_gap_expand: float = 0.48,
        tau_drift_risk: float = 0.65,
        tau_prod_filter: float = 0.65,
        tau_prod_margin: float = 0.20,
        tau_semantic_agree: float = 0.15,
    ) -> None:
        """Initialize policy with calibrated decision thresholds."""
        self.tau_clarify_prod = tau_clarify_prod
        self.tau_clarify_margin = tau_clarify_margin
        self.tau_clarify_entropy = tau_clarify_entropy
        self.tau_agree_passthrough = tau_agree_passthrough
        self.tau_gap_passthrough = tau_gap_passthrough
        self.tau_gap_expand = tau_gap_expand
        self.tau_drift_risk = tau_drift_risk
        self.tau_prod_filter = tau_prod_filter
        self.tau_prod_margin = tau_prod_margin
        self.tau_semantic_agree = tau_semantic_agree

    def decide(self, f: FeatureVector) -> PolicyDecision:
        """Evaluate feature vector and return calibrated routing decision."""
        # Rule 1: Multi-product collision with flat score distribution -> CLARIFY
        if (
            f.c_prod >= self.tau_clarify_prod
            and f.m_prod <= self.tau_clarify_margin
            and f.h_dense >= self.tau_clarify_entropy
        ):
            clarification = (
                f"Your question matches multiple product lines with similar "
                f"relevance. Are you referring to {f.top_product or 'a specific REHAU line'} "
                f"or another product material?"
            )
            return PolicyDecision(
                action=RoutingAction.CLARIFY,
                confidence=float(1.0 - f.m_prod),
                reason=(
                    f"Product collision detected (margin={f.m_prod:.2f} <= "
                    f"{self.tau_clarify_margin:.2f}, entropy={f.h_dense:.2f})"
                ),
                target_product=f.top_product,
                suggested_clarification=clarification,
            )

        # Rule 2: High lexical-dense agreement & low gap -> PASSTHROUGH
        if (
            f.j_agree >= self.tau_agree_passthrough
            and f.s_term < self.tau_gap_passthrough
        ):
            return PolicyDecision(
                action=RoutingAction.PASSTHROUGH,
                confidence=float(f.j_agree),
                reason=(
                    f"High lexical-dense concordance (Jaccard={f.j_agree:.2f} "
                    f">= {self.tau_agree_passthrough:.2f}) indicates existing keyword precision."
                ),
                target_product=f.top_product,
            )

        # Rule 3: High terminology gap & low drift risk -> EXPAND_TERMINOLOGY
        if (
            f.s_term >= self.tau_gap_expand
            and f.r_drift <= self.tau_drift_risk
        ):
            matched_term = (
                f.best_term_match.get("preferred_term", "")
                if f.best_term_match
                else ""
            )
            return PolicyDecision(
                action=RoutingAction.EXPAND_TERMINOLOGY,
                confidence=float(f.s_term),
                reason=(
                    f"Colloquial terminology gap detected (S_term={f.s_term:.2f} "
                    f">= {self.tau_gap_expand:.2f}). Safe to inject: '{matched_term}'."
                ),
                target_product=f.top_product,
            )

        # Rule 4: High product confidence & significant margin -> FILTER_METADATA
        if (
            f.c_prod >= self.tau_prod_filter
            and f.m_prod >= self.tau_prod_margin
            and f.top_product
        ):
            return PolicyDecision(
                action=RoutingAction.FILTER_METADATA,
                confidence=float(f.c_prod),
                reason=(
                    f"High product identification confidence (C_prod={f.c_prod:.2f} "
                    f">= {self.tau_prod_filter:.2f}, margin={f.m_prod:.2f}) for '{f.top_product}'."
                ),
                target_product=f.top_product,
            )

        # Rule 5: Low agreement & low terminology match -> EXPAND_SEMANTIC
        if (
            f.j_agree < self.tau_semantic_agree
            and f.s_term < self.tau_gap_expand
        ):
            return PolicyDecision(
                action=RoutingAction.EXPAND_SEMANTIC,
                confidence=float(1.0 - f.j_agree),
                reason=(
                    f"Low lexical concordance (Jaccard={f.j_agree:.2f} < "
                    f"{self.tau_semantic_agree:.2f}) requires semantic intent rewrite."
                ),
                target_product=f.top_product,
            )

        # Default fallback: PASSTHROUGH
        return PolicyDecision(
            action=RoutingAction.PASSTHROUGH,
            confidence=0.50,
            reason="Default passthrough policy (no extreme feature triggers).",
            target_product=f.top_product,
        )
