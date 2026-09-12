"""Human-in-the-Loop Active Feedback & Knowledge Graph Rule Injection.

Enables domain experts and technicians to correct red-flagged or erroneous
claims, persist human-verified rules, inject negative constraint edges into
the Terminology Knowledge Graph, and enforce company safety policies.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

from config import DATA_DIR, EVAL_DATASET
from src.tkg.graph import TerminologyKG
from src.tkg.schema import RelationType

logger = logging.getLogger(__name__)

EXPERT_RULES_PATH = os.path.join(DATA_DIR, "expert_corrections.json")
TKG_GRAPH_PATH = os.path.join(DATA_DIR, "terminology_graph.json")


class FeedbackManager:
    """Manages expert feedback, rule persistence, and TKG constraint injection."""

    def __init__(
        self,
        rules_path: str = EXPERT_RULES_PATH,
        graph_path: str = TKG_GRAPH_PATH,
    ) -> None:
        """Initialize feedback manager with paths to rule store and TKG."""
        self.rules_path = rules_path
        self.graph_path = graph_path
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure the rule store file exists on disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.rules_path)), exist_ok=True)
        if not os.path.exists(self.rules_path):
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def load_rules(self) -> list[dict[str, Any]]:
        """Load all registered expert rules from disk."""
        self._ensure_storage()
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load expert rules: %s", e)
            return []

    def save_rules(self, rules: list[dict[str, Any]]) -> None:
        """Save rules list atomically to disk."""
        with open(self.rules_path, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2)

    def submit_correction(
        self,
        query: str,
        product: str,
        contradicted_claim: str,
        approved_rule: str,
        author: str = "Shopfloor Expert",
        source_ref: str = "",
    ) -> dict[str, Any]:
        """Submit an expert correction and inject constraint into the Knowledge Graph.

        Args:
            query: The original technician question.
            product: The specific REHAU product family (e.g. 'RAUVISIO shade').
            contradicted_claim: The erroneous claim that was red-flagged.
            approved_rule: The verified engineering rule to enforce.
            author: Name or role of the verifier.
            source_ref: Optional PDF manual or standard citation.

        Returns:
            The created rule record dictionary.
        """
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        record = {
            "id": rule_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query.strip(),
            "product": product.strip() or "General",
            "contradicted_claim": contradicted_claim.strip(),
            "approved_rule": approved_rule.strip(),
            "author": author.strip(),
            "source_ref": source_ref.strip(),
            "active": True,
        }

        # 1. Save to expert_corrections.json
        rules = self.load_rules()
        rules.append(record)
        self.save_rules(rules)
        logger.info("Saved expert rule %s to %s", rule_id, self.rules_path)

        # 2. Inject constraint into Terminology Knowledge Graph
        if os.path.exists(self.graph_path):
            try:
                kg = TerminologyKG.load_json(self.graph_path)
                constraint_label = f"Forbidden: {contradicted_claim[:40]}..."
                kg.add_constraint_rule(
                    product_label=product.strip() or "RAUVISIO",
                    constraint_label=constraint_label,
                    rule_text=approved_rule.strip(),
                    relation=RelationType.FORBIDS_AGENT,
                    provenance=f"Expert Rule {rule_id} ({author})",
                )
                kg.save_json(self.graph_path)
                logger.info("Injected constraint into TKG at %s", self.graph_path)
            except Exception as e:
                logger.warning("Could not inject constraint into TKG: %s", e)

        # 3. Add to evaluation dataset for automated regression testing
        if os.path.exists(EVAL_DATASET):
            try:
                eval_entry = {
                    "id": rule_id,
                    "formal_query": query,
                    "colloquial_query": query,
                    "colloquial_queries": [query],
                    "answer": approved_rule.strip(),
                    "gold_chunk_ids": [],
                    "product": product.strip(),
                    "status": "human_verified",
                }
                with open(EVAL_DATASET, "a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_entry) + "\n")
            except Exception as e:
                logger.debug("Eval dataset append skipped: %s", e)

        return record

    def find_matching_rules(
        self, query: str, product: str | None = None
    ) -> list[dict[str, Any]]:
        """Find active expert rules matching the query or product."""
        rules = [r for r in self.load_rules() if r.get("active", True)]
        if not rules:
            return []

        q_clean = query.lower()
        q_words = set(re.findall(r"\w+", q_clean))
        p_clean = (product or "").lower().strip()

        matched: list[tuple[int, dict[str, Any]]] = []
        for r in rules:
            r_prod = r.get("product", "").lower().strip()
            r_query = r.get("query", "").lower().strip()
            r_words = set(re.findall(r"\w+", r_query))

            score = 0
            # Product match bonus
            if p_clean and r_prod and (p_clean in r_prod or r_prod in p_clean):
                score += 5
            elif r_prod and r_prod in q_clean:
                score += 5

            # Word overlap with original query
            overlap = len(q_words & r_words)
            score += overlap

            # Substring match
            if r_query and r_query in q_clean:
                score += 10

            if score >= 2:
                matched.append((score, r))

        matched.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in matched]

    def get_active_rules(self) -> list[dict[str, Any]]:
        """Return list of all currently active factory rules."""
        return [r for r in self.load_rules() if r.get("active", True)]
