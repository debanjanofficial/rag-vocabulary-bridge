"""Adaptive Router engine coordinating feature extraction, policy decision,
and strategy dispatching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.router.features import FeatureVector, extract_features
from src.router.policy import CalibratedPolicy, PolicyDecision, RoutingAction
from src.strategies.embedding_strategy import embedding_map
from src.strategies.llm_strategy import llm_map
from src.strategies.nlp_strategy import nlp_map


@dataclass
class RouterResult:
    """Consolidated outcome of the Adaptive Router execution."""

    original_query: str
    final_query: str
    action: RoutingAction
    decision: PolicyDecision
    features: FeatureVector
    retrieval_queries: list[str]
    filters: dict[str, Any] | None = None
    prefer_source: str | None = None
    clarification_prompt: str | None = None
    strategy_used: str = "Adaptive"

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "original_query": self.original_query,
            "final_query": self.final_query,
            "action": self.action.value,
            "decision": self.decision.to_dict(),
            "features": self.features.to_dict(),
            "retrieval_queries": self.retrieval_queries,
            "filters": self.filters,
            "prefer_source": self.prefer_source,
            "clarification_prompt": self.clarification_prompt,
            "strategy_used": self.strategy_used,
        }


class AdaptiveRouter:
    """Adaptive vocabulary-bridging router."""

    def __init__(self, policy: CalibratedPolicy | None = None) -> None:
        """Initialize router with a calibrated decision policy."""
        self.policy = policy or CalibratedPolicy()

    def route(
        self,
        query: str,
        llm_available: bool = True,
    ) -> RouterResult:
        """Analyze query, extract features, determine action, and dispatch."""
        q_clean = (query or "").strip()
        if not q_clean:
            empty_fv = extract_features("")
            decision = PolicyDecision(
                action=RoutingAction.PASSTHROUGH,
                confidence=1.0,
                reason="Empty query.",
            )
            return RouterResult(
                original_query="",
                final_query="",
                action=RoutingAction.PASSTHROUGH,
                decision=decision,
                features=empty_fv,
                retrieval_queries=[""],
                filters=None,
            )

        features = extract_features(q_clean)
        decision = self.policy.decide(features)

        retrieval_queries: list[str] = [q_clean]
        final_query = q_clean
        filters: dict[str, Any] | None = None
        prefer_source: str | None = None
        clarification_prompt: str | None = None

        if decision.action == RoutingAction.PASSTHROUGH:
            final_query = q_clean
            retrieval_queries = [q_clean]

        elif decision.action == RoutingAction.EXPAND_TERMINOLOGY:
            # Multi-hop Terminology Knowledge Graph (TKG) subgraph expansion
            try:
                from src.tkg import SubgraphExtractor, get_tkg
                extractor = SubgraphExtractor(get_tkg())
                subgraph_res = extractor.extract_subgraph(q_clean)
                if subgraph_res.expansion_query != q_clean:
                    final_query = subgraph_res.expansion_query
                    retrieval_queries = [final_query, q_clean]
                else:
                    nlp_res = nlp_map(q_clean)
                    retrieval_queries = nlp_res.get("retrieval_queries", [q_clean])
                    final_query = nlp_res.get("final_query", q_clean)
            except Exception:
                nlp_res = nlp_map(q_clean)
                retrieval_queries = nlp_res.get("retrieval_queries", [q_clean])
                final_query = nlp_res.get("final_query", q_clean)


        elif decision.action == RoutingAction.FILTER_METADATA:
            # Target product filtering with intent routing
            if llm_available:
                llm_res = llm_map(q_clean, llm_available=True)
                filters = llm_res.get("filters") or {}
                if not filters.get("product") and decision.target_product:
                    filters["product"] = decision.target_product
                final_query = llm_res.get("semantic_query") or q_clean
                retrieval_queries = [final_query]
                prefer_source = llm_res.get("prefer_source")
            else:
                filters = {"product": decision.target_product}
                final_query = q_clean
                retrieval_queries = [q_clean]

        elif decision.action == RoutingAction.EXPAND_SEMANTIC:
            if llm_available:
                llm_res = llm_map(q_clean, llm_available=True)
                final_query = llm_res.get("semantic_query") or q_clean
                retrieval_queries = [final_query, q_clean]
                filters = llm_res.get("filters")
                prefer_source = llm_res.get("prefer_source")
            else:
                # Rule-based expansion via NLP mapping
                nlp_res = nlp_map(q_clean)
                retrieval_queries = nlp_res.get("retrieval_queries", [q_clean])
                final_query = nlp_res.get("final_query", q_clean)

        elif decision.action == RoutingAction.CLARIFY:
            clarification_prompt = decision.suggested_clarification
            final_query = q_clean
            retrieval_queries = [q_clean]

        return RouterResult(
            original_query=q_clean,
            final_query=final_query,
            action=decision.action,
            decision=decision,
            features=features,
            retrieval_queries=retrieval_queries,
            filters=filters,
            prefer_source=prefer_source,
            clarification_prompt=clarification_prompt,
        )


_default_router: AdaptiveRouter | None = None


def get_router() -> AdaptiveRouter:
    """Return singleton AdaptiveRouter instance."""
    global _default_router
    if _default_router is None:
        _default_router = AdaptiveRouter()
    return _default_router


def route_query(query: str, llm_available: bool = True) -> RouterResult:
    """Convenience functional interface for routing a query."""
    return get_router().route(query, llm_available=llm_available)
