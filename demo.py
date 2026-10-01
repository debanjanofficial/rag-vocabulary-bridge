"""Interactive command-line demonstration for Industrial RAG Vocabulary Bridge.

Executes all 6 pillars end-to-end:
  1) 6D Query Representation Feature Extraction
  2) Calibrated Adaptive Routing
  3) Multi-Relational Terminology Knowledge Graph Subgraph Traversal
  4) Decoupled Lexical-Dense Hybrid Retrieval (RRF)
  5) Semantic Cross-Encoder Reranking
  6) Plain-Language Technical Answer Synthesis
  7) Sentence-Level Natural Language Inference (NLI) Verification
  8) Active Factory Rule Guardrail Enforcement
"""

from __future__ import annotations

import argparse
import sys
import time

from tabulate import tabulate

from src.generator import generate_answer
from src.grounding.evaluator import GroundingEvaluator
from src.reranker import rerank_documents
from src.retriever import retrieve_full_routed
from src.router.features import extract_features
from src.router.router import route_query
from src.tkg import (
    FeedbackManager,
    SafetyGuardrail,
    SubgraphExtractor,
    get_tkg,
)

PRESET_QUERIES = [
    (
        "How do I make sure the edges look good when cutting these panels?",
        "RAUVISIO ingrain - Tests sacrificial underlay terminology expansion.",
    ),
    (
        "How do I stop the laminate from bending when I press it?",
        "RAUVISIO terra - Tests core support material & warpage prevention.",
    ),
    (
        "Can I clean shade panels with window cleaner?",
        "RAUVISIO shade - Tests factory negative safety constraint & NLI.",
    ),
    (
        "What colors can these cabinet handles come in?",
        "Integrated Handles - Tests exact product matching & spec sheets.",
    ),
    (
        "How do I drill holes in the panels without breaking them?",
        "RAUVISIO crystal - Tests pilot drilling and speed guidelines.",
    ),
]


def print_banner() -> None:
    """Print executive terminal header."""
    print("=" * 79)
    print("  INDUSTRIAL RAG VOCABULARY BRIDGE - END-TO-END DEMO")
    print("  6-Pillar Knowledge-Routed Retrieval & NLI Verification")
    print("=" * 79)


def run_demo(query: str, use_llm: bool = True) -> None:
    """Run full 6-pillar pipeline demonstration on a query."""
    print(f"\n[QUERY]: \"{query}\"\n")
    t_start = time.perf_counter()

    # Pillar 1 & 2: Feature Extraction & Routing
    print("-" * 79)
    print("PILLAR 1 & 2: 6D FEATURE VECTOR & CALIBRATED ADAPTIVE ROUTING")
    print("-" * 79)
    fv = extract_features(query)
    routed = route_query(query, llm_available=use_llm)

    feat_table = [
        ["Terminology Gap Score (s_term)", f"{fv.s_term:.3f}"],
        ["Product Confidence (c_prod)", f"{fv.c_prod:.3f}"],
        ["Product Margin (m_prod)", f"{fv.m_prod:.3f}"],
        ["Dense-Lexical Agreement (j_agree)", f"{fv.j_agree:.3f}"],
        ["Dense Score Entropy (h_dense)", f"{fv.h_dense:.3f}"],
        ["Expansion Drift Risk (r_drift)", f"{fv.r_drift:.3f}"],
    ]
    print(
        tabulate(
            feat_table,
            headers=["6D Dimension", "Value"],
            tablefmt="simple",
        )
    )
    print(f"\nDecision Action:  -->  [{routed.action.value.upper()}]")
    print(f"Policy Rationale: {routed.decision.reason}")
    if routed.filters:
        print(f"Metadata Filters: {routed.filters}")

    # Pillar 3: Multi-Relational TKG Traversal
    print("\n" + "-" * 79)
    print("PILLAR 3: MULTI-RELATIONAL TERMINOLOGY KNOWLEDGE GRAPH (TKG)")
    print("-" * 79)
    tkg = get_tkg()
    extractor = SubgraphExtractor(tkg)
    subgraph_res = extractor.extract_subgraph(query)

    matched_labels = [n.label for n in subgraph_res.matched_nodes]
    matched_str = (
        str(matched_labels) if matched_labels
        else "None (Dense semantic mapping applied)"
    )
    print(f"Matched Domain Entities: {matched_str}")
    if subgraph_res.preferred_terms:
        print(f"TKG Preferred Terms:     {subgraph_res.preferred_terms}")
        print(f"Decoupled Lexical Query: \"{subgraph_res.expansion_query}\"")
        if subgraph_res.processes:
            print(f"Related Procedures:      {subgraph_res.processes}")
        if subgraph_res.linearized_paths:
            print(
                f"Multi-Hop Path Context:  {subgraph_res.get_path_summary(2)}"
            )
    else:
        print("TKG Traversal:           Direct dense mapping.")

    # Pillar 4: Decoupled Hybrid Retrieval & Reranking
    print("\n" + "-" * 79)
    print("PILLAR 4: DECOUPLED HYBRID RETRIEVAL & PATH-AWARE RERANK")
    print("-" * 79)
    candidates = retrieve_full_routed(routed, top_k=20)
    reranked = rerank_documents(
        query=query,
        docs=candidates,
        top_n=5,
        path_context=subgraph_res,
    )

    doc_rows = []
    for rank, doc in enumerate(reranked[:5], start=1):
        cid = doc.get("chunk_id", "unknown")
        src = doc.get("source", "").split("/")[-1] or doc.get("source", "")
        chan = doc.get("channel_origin", "dense_only")
        r_score = doc.get("rerank_score", doc.get("score", 0.0))
        prov = "Grounded" if doc.get("graph_provenance") else "-"
        snippet = (doc.get("document", "")[:50] + "...").replace("\n", " ")
        doc_rows.append(
            [rank, cid, src[:20], chan, prov, f"{r_score:.3f}", snippet]
        )

    headers = [
        "Rank", "Chunk ID", "Source Doc", "Channel",
        "TKG Prov", "Cross-Enc", "Passage Snippet",
    ]
    print(tabulate(doc_rows, headers=headers, tablefmt="rounded_grid"))

    # Pillar 6: Active Learning Guardrail & Deterministic Enforcement
    print("\n" + "-" * 79)
    print("PILLAR 6: FACTORY ACTIVE LEARNING & SAFETY GUARDRAILS")
    print("-" * 79)
    guardrail = SafetyGuardrail()
    matched_rules = guardrail.fm.find_matching_rules(
        query=query, product=routed.decision.target_product
    )
    if matched_rules:
        for r in matched_rules:
            print(f"  [ENFORCED RULE]: {r.get('approved_rule', '')}")
            auth = r.get("author")
            contra = r.get("contradicted_claim")
            prohib = guardrail.extract_prohibited_terms(r)
            print(
                f"  [ORIGIN]:        Author: {auth} | Contradiction: {contra}"
            )
            print(
                f"  [PROHIBITED]:    "
                f"{', '.join(prohib) if prohib else 'None'}"
            )
    else:
        print("  No active negative constraints found for this query context.")

    # Pillar 5 & Generation: Answer Synthesis & NLI Verification
    print("\n" + "-" * 79)
    print("PILLAR 5: ANSWER GENERATION & SENTENCE-LEVEL NLI GROUNDING")
    print("-" * 79)
    gen_res = generate_answer(
        query=query,
        docs=reranked,
        llm_available=use_llm,
        top_n=5,
        verify_grounding=True,
    )
    print("\n[TECHNICAL ANSWER SYNTHESIS]:")
    print(gen_res.get("answer", ""))
    print(f"\nCitations: {gen_res.get('citations') or 'None'}")
    print(f"Model Mode: {gen_res.get('model', 'Unknown')}")

    # Safety Guardrail Verification Report
    safety_info = gen_res.get("safety")
    if safety_info:
        print("\n[DETERMINISTIC SAFETY GUARDRAIL VERIFICATION]:")
        if safety_info.get("intercepted"):
            print(
                "  [CRITICAL INTERCEPTION]: Non-compliant assertion "
                "intercepted and overridden!"
            )
            v_list = safety_info.get("violations", [])
            print(f"  Active Interceptions: {len(v_list)}")
            for v in v_list:
                s_txt = v.get("violating_sentence", "")[:50]
                print(
                    f"    - [{v.get('severity')}]: Term "
                    f"'{v.get('prohibited_term')}' in \"{s_txt}...\""
                )
        else:
            s_status = "PASSED" if safety_info.get("is_safe") else "WARNING"
            score = safety_info.get("compliance_score", 1.0) * 100
            print(
                f"  Compliance Status: {s_status} (100% Policy Conformity) | "
                f"Score: {score:.0f}%"
            )

    evaluator = GroundingEvaluator()
    report = evaluator.evaluate_answer(
        answer_text=gen_res.get("answer", ""),
        docs=reranked[:5],
    )

    print("\n[SENTENCE-LEVEL NLI VERIFICATION]:")
    claim_rows = []
    for c in report.claims:
        v = c.verdict.upper()
        if "ENTAIL" in v:
            badge = "[VERIFIED / ENTAILED]"
        elif "CONTRADICT" in v:
            badge = "[HALLUCINATION / CONTRADICTION]"
        else:
            badge = "[NEUTRAL]"
        p_ent = f"{c.entailment_prob:.2f}"
        c_text = (c.text[:50] + "...") if len(c.text) > 50 else c.text
        chunk_ref = c.supporting_chunk_id or "-"
        claim_rows.append([badge, p_ent, chunk_ref, c_text])

    nli_headers = [
        "NLI Verdict", "P(Entail)", "Citing Chunk", "Factual Claim"
    ]
    print(
        tabulate(claim_rows, headers=nli_headers, tablefmt="simple")
    )
    print(
        f"\nFaithfulness: {report.faithfulness_ratio * 100:.1f}% | "
        f"Hallucination Rate: {report.hallucination_ratio * 100:.1f}%"
    )

    elapsed = time.perf_counter() - t_start
    print("=" * 79)
    print(f"Pipeline executed in {elapsed:.2f} seconds.")
    print("=" * 79)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Industrial RAG Vocabulary Bridge 6-Pillar Demo."
    )
    parser.add_argument(
        "--query", "-q", type=str, default="", help="Input technical question"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM synthesis (retrieval-only mode)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List preset industrial benchmark queries",
    )
    args = parser.parse_args()

    print_banner()

    if args.list:
        print("\nPRESET INDUSTRIAL BENCHMARK QUERIES:")
        for idx, (q, desc) in enumerate(PRESET_QUERIES, start=1):
            print(f"  [{idx}] \"{q}\"")
            print(f"      Context: {desc}")
        return

    q = args.query.strip()
    if not q:
        print("\nSelect a query to test:")
        for idx, (preset_q, desc) in enumerate(PRESET_QUERIES, start=1):
            print(f"  [{idx}] {preset_q}")
            print(f"      {desc}")
        print("  [C] Custom query entry")

        prompt_msg = "\nEnter choice [1-5 or C] (default 1): "
        choice = input(prompt_msg).strip().lower()
        if choice in ("1", "2", "3", "4", "5"):
            q = PRESET_QUERIES[int(choice) - 1][0]
        elif choice == "c":
            q = input("Enter your custom technical question: ").strip()
            if not q:
                q = PRESET_QUERIES[0][0]
        else:
            q = PRESET_QUERIES[0][0]

    run_demo(q, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
