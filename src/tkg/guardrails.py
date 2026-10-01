"""Deterministic Safety Guardrails & Rule Enforcement (Pillar 6).

Enforces zero-tolerance safety policies by validating generated responses
against active expert corrections and TKG negative constraint rules
(e.g., forbidding prohibited cleaning agents or hazardous procedures).
Intersects and overrides non-compliant advice deterministically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.tkg.feedback import FeedbackManager

logger = logging.getLogger(__name__)

# Known hazardous chemicals and abrasive agents in surface manufacturing
DEFAULT_HAZARDOUS_TERMS = {
    "window cleaner",
    "glass cleaner",
    "ammonia",
    "acetone",
    "solvent",
    "solvents",
    "abrasive",
    "abrasives",
    "steel wool",
    "scouring pad",
    "scouring powder",
    "bleach",
    "vinegar",
    "steam cleaner",
    "paint thinner",
    "lacquer thinner",
    "hydrochloric acid",
    "caustic soda",
    "benzene",
}

NEGATION_PREFIX_PATTERN = re.compile(
    r"\b(never|not|no|don't|do not|avoid|refrain from|prohibited|"
    r"forbidden|cannot|can't|should not|must not|harmful|damages?|"
    r"incompatible with)\b",
    re.IGNORECASE,
)

NEGATION_POSTFIX_PATTERN = re.compile(
    r"\b(is forbidden|is prohibited|is not allowed|is not recommended|"
    r"is unsafe|will damage|damages?|causes damage|cannot be used|"
    r"must be avoided|is banned)\b",
    re.IGNORECASE,
)


@dataclass
class SafetyViolation:
    """Individual safety policy violation detected in generated text."""

    rule_id: str
    product: str
    prohibited_term: str
    violating_sentence: str
    severity: str
    approved_rule: str

    def to_dict(self) -> dict[str, Any]:
        """Convert violation record to dictionary."""
        return asdict(self)


@dataclass
class SafetyReport:
    """Consolidated safety validation and guardrail enforcement report."""

    is_safe: bool
    violations: list[SafetyViolation] = field(default_factory=list)
    intercepted: bool = False
    original_answer: str = ""
    sanitized_answer: str = ""
    compliance_score: float = 1.0
    enforced_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert safety report to JSON-serializable dictionary."""
        return {
            "is_safe": self.is_safe,
            "violations": [v.to_dict() for v in self.violations],
            "intercepted": self.intercepted,
            "original_answer": self.original_answer,
            "sanitized_answer": self.sanitized_answer,
            "compliance_score": round(self.compliance_score, 4),
            "enforced_rules": self.enforced_rules,
        }


class SafetyGuardrail:
    """Deterministic contract validator and active safety interceptor."""

    def __init__(
        self,
        feedback_manager: FeedbackManager | None = None,
    ) -> None:
        """Initialize safety guardrail engine with feedback manager."""
        self.fm = feedback_manager or FeedbackManager()

    def extract_prohibited_terms(
        self,
        rule: dict[str, Any],
    ) -> list[str]:
        """Extract prohibited substances or actions from an expert rule."""
        terms: set[str] = set()

        claim = rule.get("contradicted_claim", "").lower()
        approved = rule.get("approved_rule", "").lower()

        # Check known hazardous vocabulary
        for hz in DEFAULT_HAZARDOUS_TERMS:
            if hz in claim or hz in approved:
                terms.add(hz)

        # Pattern match explicit prohibitions: "Never use X", "Do not use X"
        matches = re.findall(
            r"(?:never use|do not use|avoid|prohibit)\s+([a-z0-9\s\-]+?)"
            r"(?:\s+on|\s+diluted|\s+for|[.,;]|$)",
            approved,
            re.IGNORECASE,
        )
        for m in matches:
            cleaned = m.strip().lower()
            if 3 <= len(cleaned) <= 30:
                terms.add(cleaned)

        return sorted(list(terms))

    def is_negated_or_safe(
        self,
        sentence: str,
        term: str,
    ) -> bool:
        """Return True if prohibited term is explicitly negated or warned."""
        s_low = sentence.lower()
        t_low = term.lower()

        if t_low not in s_low:
            return True

        # Check prefix negation (e.g., "Do not use window cleaner")
        term_idx = s_low.find(t_low)
        prefix_window = s_low[max(0, term_idx - 45):term_idx]
        if NEGATION_PREFIX_PATTERN.search(prefix_window):
            return True

        # Check postfix negation (e.g., "Window cleaner is forbidden")
        end_idx = term_idx + len(t_low)
        postfix_window = s_low[end_idx : end_idx + 45]
        if NEGATION_POSTFIX_PATTERN.search(postfix_window):
            return True

        return False

    def validate_answer(
        self,
        answer: str,
        query: str,
        product: str | None = None,
        auto_sanitize: bool = True,
    ) -> SafetyReport:
        """Validate generated text against active factory rules and TKG.

        Args:
            answer: Raw generated text from model or retrieval fallback.
            query: Inbound user question string.
            product: Active product line (e.g. 'RAUVISIO shade').
            auto_sanitize: If True, deterministically rewrites violations.

        Returns:
            SafetyReport containing compliance metrics and sanitized text.
        """
        if not answer or not answer.strip():
            return SafetyReport(
                is_safe=True,
                original_answer=answer,
                sanitized_answer=answer,
                compliance_score=1.0,
            )

        active_rules = self.fm.find_matching_rules(query, product)
        if not active_rules:
            return SafetyReport(
                is_safe=True,
                original_answer=answer,
                sanitized_answer=answer,
                compliance_score=1.0,
            )

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", answer)
            if s.strip()
        ]

        violations: list[SafetyViolation] = []
        enforced_rule_texts: list[str] = []
        violating_sentences: set[str] = set()

        for rule in active_rules:
            rule_id = rule.get("id", "rule_unknown")
            r_prod = rule.get("product", product or "General")
            approved = rule.get("approved_rule", "")
            prohibited_terms = self.extract_prohibited_terms(rule)

            for sentence in sentences:
                for term in prohibited_terms:
                    if term.lower() in sentence.lower():
                        if not self.is_negated_or_safe(sentence, term):
                            violation = SafetyViolation(
                                rule_id=rule_id,
                                product=r_prod,
                                prohibited_term=term,
                                violating_sentence=sentence,
                                severity="CRITICAL",
                                approved_rule=approved,
                            )
                            violations.append(violation)
                            violating_sentences.add(sentence)
                            if approved not in enforced_rule_texts:
                                enforced_rule_texts.append(approved)

        is_safe = len(violations) == 0
        compliance_score = max(0.0, 1.0 - (len(violations) * 0.5))

        sanitized = answer
        intercepted = False

        if not is_safe and auto_sanitize:
            intercepted = True
            # Build clean safety override instruction
            override_msg = "\n".join(
                f"[MANDATORY FACTORY SAFETY OVERRIDE]: {r}"
                for r in enforced_rule_texts
            )

            # Filter out violating sentences from the answer
            retained_sentences = [
                s for s in sentences if s not in violating_sentences
            ]

            if not retained_sentences:
                sanitized = override_msg
            else:
                body = " ".join(retained_sentences)
                sanitized = f"{override_msg}\n\n{body}"

            logger.warning(
                "Safety Guardrail INTERCEPTED %d safety violation(s).",
                len(violations),
            )

        return SafetyReport(
            is_safe=is_safe,
            violations=violations,
            intercepted=intercepted,
            original_answer=answer,
            sanitized_answer=sanitized,
            compliance_score=compliance_score,
            enforced_rules=enforced_rule_texts,
        )


_default_guardrail: SafetyGuardrail | None = None


def get_safety_guardrail() -> SafetyGuardrail:
    """Return singleton instance of SafetyGuardrail."""
    global _default_guardrail
    if _default_guardrail is None:
        _default_guardrail = SafetyGuardrail()
    return _default_guardrail
