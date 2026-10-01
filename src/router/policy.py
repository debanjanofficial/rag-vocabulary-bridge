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
import json
import os
from typing import Any

import numpy as np

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
    action_probabilities: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to a JSON-serializable dictionary."""
        data = {
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "target_product": self.target_product,
            "suggested_clarification": self.suggested_clarification,
        }
        if self.action_probabilities is not None:
            data["action_probabilities"] = {
                k: round(v, 4) for k, v in self.action_probabilities.items()
            }
        return data


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
        # Rule 1: Multi-product collision with flat distribution -> CLARIFY
        if (
            f.c_prod >= self.tau_clarify_prod
            and f.m_prod <= self.tau_clarify_margin
            and f.h_dense >= self.tau_clarify_entropy
        ):
            prod_name = f.top_product or "a specific REHAU line"
            clarification = (
                f"Your question matches multiple product lines with similar "
                f"relevance. Are you referring to {prod_name} or another "
                f"product material?"
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
                    f">= {self.tau_agree_passthrough:.2f}) indicates keyword "
                    f"precision."
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
                    f"Colloquial terminology gap detected "
                    f"(S_term={f.s_term:.2f} >= {self.tau_gap_expand:.2f}). "
                    f"Safe to inject: '{matched_term}'."
                ),
                target_product=f.top_product,
            )

        # Rule 4: High product confidence & margin -> FILTER_METADATA
        if (
            f.c_prod >= self.tau_prod_filter
            and f.m_prod >= self.tau_prod_margin
            and f.top_product
        ):
            return PolicyDecision(
                action=RoutingAction.FILTER_METADATA,
                confidence=float(f.c_prod),
                reason=(
                    f"High product identification confidence "
                    f"(C_prod={f.c_prod:.2f} >= {self.tau_prod_filter:.2f}, "
                    f"margin={f.m_prod:.2f}) for '{f.top_product}'."
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
                    f"{self.tau_semantic_agree:.2f}) requires semantic "
                    f"intent rewrite."
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


class LearnedGatingPolicy:
    """Probabilistic learned gating policy for adaptive query routing.

    Evaluates the 6D feature vector Phi(q) through trained multinomial
    logistic regression weights W in R^(5x6) and intercepts b in R^5:
        z = W * Phi(q) + b
        P(action | Phi(q)) = Softmax(z)

    Yields calibrated action posterior probabilities, interpretable
    feature contribution drivers, and adaptive action dispatching.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        fallback_policy: CalibratedPolicy | None = None,
    ) -> None:
        """Initialize learned gating policy and load model coefficients."""
        self.fallback = fallback_policy or CalibratedPolicy()
        if weights_path is not None:
            self.weights_path = weights_path
        else:
            self.weights_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "learned_weights.json",
            )
        self.action_names = [
            RoutingAction.PASSTHROUGH.value,
            RoutingAction.EXPAND_TERMINOLOGY.value,
            RoutingAction.FILTER_METADATA.value,
            RoutingAction.EXPAND_SEMANTIC.value,
            RoutingAction.CLARIFY.value,
        ]
        self._load_weights()

    def _load_weights(self) -> None:
        """Load weight matrix and intercept vector from JSON file."""
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.weights = np.array(data["weights"], dtype=float)
                self.intercept = np.array(data["intercept"], dtype=float)
                self.action_names = data.get("action_names", self.action_names)
                return
            except Exception:
                pass

        # Calibrated default coefficients (trained on boundary distributions)
        self.weights = np.array([
            [-2.095, -2.322, -0.342, +7.792, -2.883, -0.834],
            [+8.848, -2.034, +0.861, -0.411, +1.024, +1.485],
            [-1.302, +6.251, +5.234, +0.144, -1.591, -0.108],
            [-3.932, -5.125, -0.654, -5.669, -2.707, +0.390],
            [-1.518, +3.230, -5.100, -1.856, +6.157, -0.934],
        ], dtype=float)
        self.intercept = np.array([
            +1.882, -2.890, -2.943, +6.901, -2.949
        ], dtype=float)

    def decide(self, f: FeatureVector) -> PolicyDecision:
        """Evaluate features through learned gating network."""
        try:
            x = f.to_array()
            logits = self.weights @ x + self.intercept
            exp_z = np.exp(logits - np.max(logits))
            probs = exp_z / np.sum(exp_z)

            best_idx = int(np.argmax(probs))
            action_str = self.action_names[best_idx]
            action = RoutingAction(action_str)
            confidence = float(probs[best_idx])

            prob_dict = {
                self.action_names[i]: float(probs[i])
                for i in range(len(probs))
            }

            feat_names = [
                "S_term", "C_prod", "M_prod",
                "J_agree", "H_dense", "R_drift",
            ]
            contrib = self.weights[best_idx] * x
            top_idx = int(np.argmax(np.abs(contrib)))
            top_feat = feat_names[top_idx]
            val = x[top_idx]

            reason = (
                f"Learned Gating Model selected {action.value.upper()} with "
                f"{confidence * 100:.1f}% posterior probability "
                f"(primary driver: {top_feat}={val:.2f})."
            )

            clarification = None
            if action == RoutingAction.CLARIFY:
                clarification = (
                    f"Your question matches multiple product lines with "
                    f"similar relevance. Are you referring to "
                    f"{f.top_product or 'a specific REHAU line'} "
                    f"or another product material?"
                )

            return PolicyDecision(
                action=action,
                confidence=confidence,
                reason=reason,
                target_product=f.top_product,
                suggested_clarification=clarification,
                action_probabilities=prob_dict,
            )
        except Exception:
            return self.fallback.decide(f)
