"""Evaluation benchmark for Answer Grounding & Sentence-Level NLI Citations (Pillar 5).

Evaluates Faithfulness (Grounding Ratio), Hallucination Rate, Citation
Precision/Recall, and TKG Provenance alignment across evaluation dataset queries.
Outputs publication-ready markdown and JSON benchmarks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

from config import EVAL_DATASET, RESULTS_DIR
from src.grounding.evaluator import GroundingEvaluator
from src.retriever import retrieve_full, retrieve_full_routed
from src.router.router import AdaptiveRouter
from src.tkg.graph import TerminologyKG
from src.tkg.provenance import ProvenanceTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_eval_data(limit: int | None = None) -> list[dict[str, Any]]:
    """Load evaluation samples from eval_dataset.jsonl."""
    if not os.path.exists(EVAL_DATASET):
        logger.error("Evaluation dataset not found at %s", EVAL_DATASET)
        return []

    samples = []
    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def evaluate_grounding_benchmark(
    samples: list[dict[str, Any]],
    evaluator: GroundingEvaluator,
    tkg_tracker: ProvenanceTracker | None = None,
    router: AdaptiveRouter | None = None,
) -> dict[str, Any]:
    """Run comparative grounding evaluation between Baseline and Adaptive pipelines."""
    strategies = ["Baseline", "Adaptive (Auto-Route)"]
    metrics: dict[str, dict[str, Any]] = {
        s: {
            "faithfulness": [],
            "hallucination": [],
            "citation_precision": [],
            "tkg_provenance": [],
            "num_claims": [],
        }
        for s in strategies
    }

    logger.info("Evaluating grounding on %d queries...", len(samples))

    for idx, sample in enumerate(samples, 1):
        query = sample.get("colloquial_query") or sample.get("query", "")
        gold_answer = sample.get("answer", "")

        logger.info(
            "[%d/%d] Query: %s",
            idx,
            len(samples),
            query[:60] + "..." if len(query) > 60 else query,
        )

        for strat in strategies:
            if strat == "Baseline":
                docs = retrieve_full(query, top_k=5)
            else:
                if router:
                    routed_res = router.route(query)
                    docs = retrieve_full_routed(routed_res.to_dict(), top_k=5)
                else:
                    docs = retrieve_full(query, top_k=5)

            # Evaluate answer grounding against retrieved docs
            report = evaluator.evaluate_answer(
                answer_text=gold_answer,
                docs=docs,
                tkg_tracker=tkg_tracker,
            )

            metrics[strat]["faithfulness"].append(report.faithfulness_ratio)
            metrics[strat]["hallucination"].append(report.hallucination_ratio)
            metrics[strat]["citation_precision"].append(report.citation_precision)
            metrics[strat]["tkg_provenance"].append(report.tkg_provenance_ratio)
            metrics[strat]["num_claims"].append(report.num_claims)

    summary: dict[str, Any] = {}
    for strat, data in metrics.items():
        n = len(data["faithfulness"]) or 1
        summary[strat] = {
            "mean_faithfulness_pct": round(sum(data["faithfulness"]) / n * 100, 2),
            "mean_hallucination_pct": round(sum(data["hallucination"]) / n * 100, 2),
            "mean_citation_precision_pct": round(
                sum(data["citation_precision"]) / n * 100, 2
            ),
            "mean_tkg_provenance_pct": round(
                sum(data["tkg_provenance"]) / n * 100, 2
            ),
            "total_claims_evaluated": sum(data["num_claims"]),
            "evaluated_queries": n,
        }

    return summary


def main() -> None:
    """Entry point for CLI grounding evaluation."""
    parser = argparse.ArgumentParser(
        description="Benchmark Answer Grounding & Sentence-Level NLI Citations"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of queries to evaluate (default: 10)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(RESULTS_DIR, "evaluation_results_grounding.json"),
        help="Output path for evaluation JSON results",
    )
    args = parser.parse_args()

    samples = load_eval_data(limit=args.limit)
    if not samples:
        logger.error("No samples found.")
        sys.exit(1)

    logger.info("Initializing GroundingEvaluator and Terminology KG...")
    evaluator = GroundingEvaluator()

    tkg_tracker = None
    graph_path = os.path.join("data", "terminology_graph.json")
    if os.path.exists(graph_path):
        kg = TerminologyKG.load_json(graph_path)
        tkg_tracker = ProvenanceTracker(kg)
        logger.info("Loaded TKG with %d nodes for provenance binding.", kg.node_count())

    router = AdaptiveRouter()

    t0 = time.time()
    results = evaluate_grounding_benchmark(
        samples=samples,
        evaluator=evaluator,
        tkg_tracker=tkg_tracker,
        router=router,
    )
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "summary": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved grounding evaluation results to %s", args.out)

    print("\n" + "=" * 70)
    print("        PILLAR 5: ANSWER GROUNDING & NLI CITATION BENCHMARK        ")
    print("=" * 70)
    print(
        f"{'Strategy':<25} | {'Faithfulness':<12} | {'Hallucination':<14} | {'Precision':<10} | {'TKG Provenance':<14}"
    )
    print("-" * 70)
    for strat, res in results.items():
        print(
            f"{strat:<25} | {res['mean_faithfulness_pct']:>10.1f}% | {res['mean_hallucination_pct']:>12.1f}% | {res['mean_citation_precision_pct']:>8.1f}% | {res['mean_tkg_provenance_pct']:>12.1f}%"
        )
    print("=" * 70)
    print(f"Elapsed: {elapsed:.2f}s | Evaluated {len(samples)} queries\n")


if __name__ == "__main__":
    main()
