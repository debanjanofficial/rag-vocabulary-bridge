"""Adaptive Terminology-Grounded Router package.

Provides feature extraction, calibrated risk policies, and routing
coordination for bridging lay-expert vocabulary gaps in industrial RAG.
"""

from __future__ import annotations

from src.router.bm25_index import BM25Index, get_bm25_index
from src.router.features import (
    FeatureVector,
    compute_expansion_drift_risk,
    compute_lexical_dense_agreement,
    compute_product_confidence,
    compute_retrieval_entropy,
    compute_terminology_gap,
    extract_features,
)
from src.router.policy import (
    CalibratedPolicy,
    PolicyDecision,
    RoutingAction,
)
from src.router.router import (
    AdaptiveRouter,
    RouterResult,
    get_router,
    route_query,
)

__all__ = [
    "AdaptiveRouter",
    "BM25Index",
    "CalibratedPolicy",
    "FeatureVector",
    "PolicyDecision",
    "RouterResult",
    "RoutingAction",
    "compute_expansion_drift_risk",
    "compute_lexical_dense_agreement",
    "compute_product_confidence",
    "compute_retrieval_entropy",
    "compute_terminology_gap",
    "extract_features",
    "get_bm25_index",
    "get_router",
    "route_query",
]
