"""Claim extraction and sentence decomposition for grounding verification.

Decomposes generated answers into atomic verifiable claims while preserving
technical standards, decimal quantities, and abbreviations, and filtering out
conversational scaffolding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimStatement:
    """Atomic claim extracted from a generated answer."""

    claim_id: int
    text: str
    original_text: str
    is_factual: bool = True


# Abbreviations and technical tokens where period should not split sentences.
_ABBREVIATIONS = (
    r"e\.g\.",
    r"i\.e\.",
    r"approx\.",
    r"min\.",
    r"max\.",
    r"fig\.",
    r"no\.",
    r"vs\.",
    r"dept\.",
    r"art\.",
    r"ref\.",
    r"co\.",
    r"ltd\.",
    r"inc\.",
    r"din\s+\d+",
    r"en\s+\d+",
    r"astm\s+[a-z0-9]+",
    r"iso\s+\d+",
)

_SCAFFOLDING_PATTERNS = (
    r"^sources?\s*:\s*.*$",
    r"^source\s+passages?\s*:.*$",
    r"^here\s+is\s+the\s+.*:?$",
    r"^based\s+on\s+the\s+.*:?$",
    r"^according\s+to\s+the\s+.*:?$",
    r"^in\s+summary\s*,?:?$",
    r"^to\s+answer\s+your\s+question\s*,?:?$",
    r"^please\s+note\s+that\s*:?$",
    r"^note\s*:?$",
)


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    """Replace technical abbreviations with placeholders to prevent bad splits."""
    mapping: dict[str, str] = {}
    protected = text

    for i, pattern in enumerate(_ABBREVIATIONS):
        matches = list(re.finditer(pattern, protected, flags=re.IGNORECASE))
        for match in reversed(matches):
            token = match.group(0)
            placeholder = f"__ABBR_{i}_{len(mapping)}__"
            mapping[placeholder] = token
            protected = (
                protected[: match.start()]
                + placeholder
                + protected[match.end() :]
            )

    # Also protect decimal numbers like 2.5 mm or 0.8 mm
    decimal_matches = list(re.finditer(r"\b\d+\.\d+\b", protected))
    for match in reversed(decimal_matches):
        token = match.group(0)
        placeholder = f"__DECIMAL_{len(mapping)}__"
        mapping[placeholder] = token
        protected = (
            protected[: match.start()] + placeholder + protected[match.end() :]
        )

    return protected, mapping


def _restore_abbreviations(text: str, mapping: dict[str, str]) -> str:
    """Restore original tokens from placeholders."""
    restored = text
    for placeholder, original in mapping.items():
        restored = restored.replace(placeholder, original)
    return restored


def _is_scaffolding(text: str) -> bool:
    """Detect whether a line is non-factual scaffolding or citation footer."""
    stripped = text.strip()
    if not stripped:
        return True

    low = stripped.lower()
    for pattern in _SCAFFOLDING_PATTERNS:
        if re.match(pattern, low):
            return True

    # Check if line is purely bracketed citations or empty bullets
    if re.fullmatch(r"^[\[\d\],\s\-*•]+$", stripped):
        return True

    return False


def extract_claims(answer_text: str) -> list[ClaimStatement]:
    """Decompose answer into atomic factual claim statements.

    Splits text across newlines, bullet points, and sentence boundaries while
    preserving technical specifications and ignoring conversational headers
    or citation footers.
    """
    if not answer_text or not answer_text.strip():
        return []

    lines = answer_text.splitlines()
    raw_candidates: list[str] = []

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line or _is_scaffolding(cleaned_line):
            continue

        # Strip bullet prefixes (- , * , 1. , etc.)
        cleaned_line = re.sub(r"^(\s*[-*•]|\s*\d+[\.)])\s+", "", cleaned_line)

        # Protect abbreviations and split on sentence terminators (. ! ?)
        protected, mapping = _protect_abbreviations(cleaned_line)
        splits = re.split(r"(?<=[.!?])\s+", protected)

        for s in splits:
            restored = _restore_abbreviations(s, mapping).strip()
            if restored and not _is_scaffolding(restored):
                raw_candidates.append(restored)

    claims: list[ClaimStatement] = []
    claim_idx = 1

    for raw in raw_candidates:
        # Strip trailing citation markers like [1], [2], or (chunk_id)
        cleaned_claim = re.sub(r"\[\s*\d+\s*\]", "", raw).strip()
        cleaned_claim = re.sub(r"\(\s*doc_\d+\s*\)", "", cleaned_claim).strip()
        cleaned_claim = re.sub(r"\s+", " ", cleaned_claim)

        if len(cleaned_claim) < 5:
            continue

        # Check factual predicate: contains word characters and not purely greeting
        is_factual = bool(
            re.search(r"[a-zA-Z]{2,}", cleaned_claim)
            and not cleaned_claim.lower().startswith("thank")
        )

        claims.append(
            ClaimStatement(
                claim_id=claim_idx,
                text=cleaned_claim,
                original_text=raw,
                is_factual=is_factual,
            )
        )
        claim_idx += 1

    return claims
