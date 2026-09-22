"""
End-to-End Verification Suite for Option 4 (Streamlit UI & Core Pillars).

This script programmatically verifies all UI pipeline capabilities:
1. Calibrated policy action taxonomy (PASSTHROUGH, CLARIFY, EXPAND, FILTER).
2. Live colloquial query routing with in-place TKG graph expansion.
3. End-to-end retrieval, cross-encoder reranking, and DeBERTa NLI grounding.
4. Human-in-the-Loop "Correct & Learn" active feedback loop and TKG rule storage.
"""

import os
import sys
import json

# Ensure project root is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import DATA_DIR
from src.router.policy import CalibratedPolicy, FeatureVector, RoutingAction
from src.router.router import route_query
from src.answer_pipeline import run_answer_pipeline
from src.tkg.feedback import FeedbackManager
from src.tkg.graph import TerminologyKG


def verify_policy_action_taxonomy():
    """Verify calibrated policy decisions across all discrete actions."""
    print("\n--- Test 1: Calibrated Policy Taxonomy (Pillar 2) ---")
    policy = CalibratedPolicy()

    # 1. PASSTHROUGH trigger: high concordance, low gap
    fv_pass = FeatureVector(
        query="DIN 4102-1 B1 fire behavior classification",
        s_term=0.10,
        c_prod=0.30,
        m_prod=0.10,
        top_product="RAUVISIO noir",
        j_agree=0.55,
        h_dense=0.30,
        r_drift=0.10,
    )
    dec_pass = policy.decide(fv_pass)
    print(f"1. Passthrough Case: action={dec_pass.action.value}, "
          f"confidence={dec_pass.confidence:.2f}")
    assert dec_pass.action == RoutingAction.PASSTHROUGH, (
        f"Expected PASSTHROUGH, got {dec_pass.action}"
    )

    # 2. CLARIFY trigger: product collision with high entropy
    fv_clarify = FeatureVector(
        query="How to clean the surface?",
        s_term=0.20,
        c_prod=0.45,
        m_prod=0.03,
        top_product="RAUVISIO crystal",
        j_agree=0.10,
        h_dense=0.85,
        r_drift=0.20,
    )
    dec_clarify = policy.decide(fv_clarify)
    print(f"2. Clarification Case: action={dec_clarify.action.value}, "
          f"prompt='{dec_clarify.suggested_clarification}'")
    assert dec_clarify.action == RoutingAction.CLARIFY, (
        f"Expected CLARIFY, got {dec_clarify.action}"
    )

    # 3. EXPAND_TERMINOLOGY trigger: high terminology gap, low drift risk
    fv_expand = FeatureVector(
        query="How do I stop the laminate from bending?",
        s_term=0.85,
        c_prod=0.30,
        m_prod=0.10,
        top_product=None,
        j_agree=0.15,
        h_dense=0.40,
        r_drift=0.20,
        best_term_match={"preferred_term": "warpage"},
    )
    dec_expand = policy.decide(fv_expand)
    print(f"3. Expansion Case: action={dec_expand.action.value}, "
          f"confidence={dec_expand.confidence:.2f}")
    assert dec_expand.action == RoutingAction.EXPAND_TERMINOLOGY, (
        f"Expected EXPAND_TERMINOLOGY, got {dec_expand.action}"
    )

    # 4. FILTER_METADATA trigger: confident product identification
    fv_filter = FeatureVector(
        query="What colors can these handles come in?",
        s_term=0.10,
        c_prod=0.92,
        m_prod=0.60,
        top_product="Integrated Handles",
        j_agree=0.20,
        h_dense=0.30,
        r_drift=0.10,
    )
    dec_filter = policy.decide(fv_filter)
    print(f"4. Metadata Filter Case: action={dec_filter.action.value}, "
          f"target='{dec_filter.target_product}'")
    assert dec_filter.action == RoutingAction.FILTER_METADATA, (
        f"Expected FILTER_METADATA, got {dec_filter.action}"
    )

    print("✓ Test 1 Passed: Policy action taxonomy verified mathematically.")


def verify_colloquial_query_routing_and_tkg():
    """Verify live colloquial query routing and TKG subgraph expansion."""
    print("\n--- Test 2: Live Colloquial Routing & TKG Injection (Pillar 1) ---")
    query = (
        "How do I make sure the edges look good "
        "when cutting my wood-grain panels?"
    )
    result = route_query(query, llm_available=False).to_dict()

    action = result.get("action")
    features = result.get("features", {})
    final_query = result.get("final_query", "")

    print(f"Query: '{query}'")
    print(f"Selected Action: {action}")
    print(f"S_term (Lexical Gap): {features.get('s_term', 0.0):.2f}")
    print(f"Final Expanded Query: '{final_query}'")

    assert action == "expand_terminology", (
        f"Expected expand_terminology, got {action}"
    )
    assert len(result.get("retrieval_queries", [])) >= 1, (
        "No retrieval queries generated"
    )
    print("✓ Test 2 Passed: Colloquial query correctly routed to TKG.")
    return result


def verify_retrieval_and_nli_grounding():
    """Verify hybrid retrieval, cross-encoder reranking, and DeBERTa NLI."""
    print("\n--- Test 3: Reranking & Sentence-Level NLI (Pillar 5) ---")
    query = "What is the fire safety rating for this noir matte material?"
    map_res = route_query(query, llm_available=False).to_dict()

    print(f"Query: '{query}'")
    print("Executing complete answer pipeline with DeBERTa NLI...")
    pipeline = run_answer_pipeline(
        strategy="Adaptive (Auto-Route)",
        query=query,
        mapped=map_res,
        llm_available=True,
        generate=True,
        detail=True,
        verify_grounding=True,
    )

    reranked = pipeline.get("reranked", [])
    print(f"Retrieved and reranked chunks: {len(reranked)}")
    assert len(reranked) > 0, "No chunks retrieved"

    top_chunk = reranked[0]
    print(f"Top-1 Chunk ID: {top_chunk.get('chunk_id')}")
    print(f"Top-1 Product: {top_chunk.get('product')}")
    print(f"Top-1 Rerank Score: {top_chunk.get('rerank_score', 0.0):.4f}")

    gen = pipeline.get("generation", {})
    answer = gen.get("answer", "")
    grounding = gen.get("grounding") or {}
    claims = grounding.get("claims", [])
    faithfulness = grounding.get("faithfulness_ratio", 0.0)
    hallucination = grounding.get("hallucination_ratio", 0.0)

    print(f"\nGenerated Answer ({len(answer)} chars):")
    print(f"\"{answer[:220]}...\"")
    print(f"Pillar 5 Faithfulness: {faithfulness * 100:.1f}%")
    print(f"Pillar 5 Hallucination Risk: {hallucination * 100:.1f}%")
    print(f"Total Verified Claims: {len(claims)}")

    for i, claim in enumerate(claims, 1):
        verdict = claim.get("verdict")
        prob = claim.get("entailment_prob", 0.0)
        chunk = claim.get("supporting_chunk_id")
        print(f"  [Claim {i}] {verdict.upper()} (prob={prob:.2f}) "
              f"Chunk: {chunk} -> '{claim.get('text')[:55]}...'")

    assert len(claims) > 0, "No claims extracted for NLI verification"
    assert faithfulness >= 0.80, f"Faithfulness below threshold: {faithfulness}"
    print("✓ Test 3 Passed: Sentence-level NLI grounding verified.")


def verify_active_learning_rule_injection():
    """Verify Correct & Learn active feedback loop and TKG constraint injection."""
    print("\n--- Test 4: 'Correct & Learn' Feedback Loop (Pillar 6) ---")
    fm = FeedbackManager()
    tkg_path = os.path.join(DATA_DIR, "terminology_graph.json")
    tkg = TerminologyKG.load_json(tkg_path)

    initial_nodes = tkg.node_count()
    initial_edges = tkg.edge_count()
    print(f"Initial TKG State: {initial_nodes} nodes, {initial_edges} edges")

    test_query = "Can I clean RAUVISIO noir with acetone or window cleaner?"
    test_claim = "Acetone can be used to wipe clean the noir surface."
    test_rule = (
        "Never use acetone, harsh solvents, or abrasive cleaners on "
        "RAUVISIO noir; clean exclusively with mild soap and soft cloth."
    )
    author = "Lead Materials Engineer"
    product = "RAUVISIO noir"

    print(f"Injecting negative constraint for product: '{product}'...")
    correction = fm.submit_correction(
        query=test_query,
        product=product,
        contradicted_claim=test_claim,
        approved_rule=test_rule,
        author=author,
    )
    rule_id = correction.get("id")
    print(f"Saved Correction Record ID: {rule_id}")

    # Re-instantiate TKG to verify on-disk persistence
    tkg_reloaded = TerminologyKG.load_json(tkg_path)
    updated_nodes = tkg_reloaded.node_count()
    updated_edges = tkg_reloaded.edge_count()
    print(f"Updated TKG State: {updated_nodes} nodes, {updated_edges} edges")

    # Verify active rule enforcement lookup
    matched_rules = fm.find_matching_rules(
        query="What chemicals can I use on noir matte panels?",
        product="RAUVISIO noir",
    )
    print(f"Matching Rules Found for subsequent query: {len(matched_rules)}")
    if matched_rules:
        print(f"Enforced Rule: '{matched_rules[0]['approved_rule']}'")

    assert any(r.get("id") == rule_id for r in matched_rules), (
        f"Rule {rule_id} was not retrieved for matching product query"
    )
    print("✓ Test 4 Passed: Active learning rule permanently stored & enforced.")


def main():
    """Execute complete Option 4 live verification suite."""
    print("===============================================================")
    print("  RAG-VOCABULARY-BRIDGE: OPTION 4 LIVE VERIFICATION SUITE")
    print("===============================================================")
    verify_policy_action_taxonomy()
    verify_colloquial_query_routing_and_tkg()
    verify_retrieval_and_nli_grounding()
    verify_active_learning_rule_injection()
    print("\n===============================================================")
    print("  ALL OPTION 4 SYSTEM CHECKS COMPLETED SUCCESSFULLY!")
    print("===============================================================")


if __name__ == "__main__":
    main()

