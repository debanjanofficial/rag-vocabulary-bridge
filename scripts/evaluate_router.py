"""Scientific evaluation and benchmark comparison of Adaptive Router vs baselines.

Evaluates:
  1) Baseline (Dense-only, no mapping)
  2) NLP Mapping (Rule-based phrase injection + RRF)
  3) Embedding Mapping (Synonym cosine matching + RRF)
  4) LLM Self-Query (Intent parsing + hard metadata filtering)
  5) Adaptive Router (Confidence-aware risk-controlled policy)

Metrics:
  - Recall@1, Recall@3, Recall@5
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG@5)
  - Mean Query Latency (ms)
  - Action Distribution (%)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import EVAL_DATASET, RESULTS_DIR
from src.retriever import (
    retrieve,
    retrieve_routed,
    retrieve_rrf,
    retrieve_self_query,
)
from src.router.router import route_query
from src.strategies.embedding_strategy import embedding_map
from src.strategies.llm_strategy import llm_map
from src.strategies.nlp_strategy import nlp_map

OUT_RESULTS_JSON = os.path.join(RESULTS_DIR, "evaluation_results_router.json")


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Compute Recall@k for binary relevance."""
    return 1.0 if any(r in gold for r in retrieved[:k]) else 0.0


def mrr(retrieved: list[str], gold: set[str]) -> float:
    """Compute Reciprocal Rank (RR) for the first relevant document."""
    for i, rid in enumerate(retrieved):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Compute normalized Discounted Cumulative Gain at rank k."""
    for i, rid in enumerate(retrieved[:k]):
        if rid in gold:
            return 1.0 / math.log2(i + 2)
    return 0.0


def run_baseline(q: str, _llm: bool, k: int) -> list[str]:
    """Execute baseline dense retrieval without query reformulation."""
    return retrieve(q, top_k=k)


def run_nlp(q: str, _llm: bool, k: int) -> list[str]:
    """Execute NLP phrase-gated mapping + RRF."""
    mapped = nlp_map(q)
    queries = mapped.get("retrieval_queries") or [q]
    return retrieve_rrf(queries, top_k=k)


def run_embedding(q: str, _llm: bool, k: int) -> list[str]:
    """Execute embedding synonym mapping + RRF."""
    mapped = embedding_map(q)
    queries = mapped.get("retrieval_queries") or [mapped.get("final_query", q)]
    return retrieve_rrf(queries, top_k=k)


def run_llm_self_query(q: str, llm: bool, k: int) -> list[str]:
    """Execute LLM Self-Query with metadata filtering."""
    mapped = llm_map(q, llm_available=llm)
    filters = mapped.get("filters") or {}
    return retrieve_self_query(
        original_query=mapped.get("original_query") or q,
        semantic_query=mapped.get("semantic_query") or q,
        product=filters.get("product"),
        doc_type=filters.get("doc_type"),
        prefer_source=mapped.get("prefer_source"),
        top_k=k,
    )


def run_adaptive_router(q: str, llm: bool, k: int) -> tuple[list[str], Any]:
    """Execute Adaptive Router feature extraction, policy, and retrieval."""
    router_res = route_query(q, llm_available=llm)
    retrieved_ids = retrieve_routed(router_res, top_k=k)
    return retrieved_ids, router_res


def evaluate_all(
    llm: bool = True,
    top_k_vals: tuple[int, ...] = (1, 3, 5),
    limit: int = 0,
    out_path: str = OUT_RESULTS_JSON,
) -> dict[str, Any]:
    """Run end-to-end evaluation across all strategies and compile report."""
    if not os.path.exists(EVAL_DATASET):
        raise FileNotFoundError(f"{EVAL_DATASET} not found.")

    with open(EVAL_DATASET, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if limit > 0:
        items = items[:limit]

    max_k = max(top_k_vals)
    print(f"\n=======================================================")
    print(f"Starting Evaluation on {len(items)} questions (LLM={llm})")
    print(f"=======================================================\n")

    strategies: dict[str, Any] = {
        "Baseline": run_baseline,
        "NLP Mapping": run_nlp,
        "Embedding Mapping": run_embedding,
    }
    if llm:
        strategies["LLM Self-Query"] = run_llm_self_query

    all_results: dict[str, Any] = {}

    # Run standard baselines
    for name, fn in strategies.items():
        print(f"Benchmarking: {name} ...", end="", flush=True)
        k_scores = {k: [] for k in top_k_vals}
        mrr_list: list[float] = []
        ndcg_list: list[float] = []
        latencies: list[float] = []

        t_start = time.time()
        for item in items:
            q = item["colloquial_query"]
            gold = set(item["gold_chunk_ids"])

            t0 = time.time()
            try:
                retrieved = fn(q, llm, max_k)
            except Exception as exc:
                print(f" [Error: {exc}]", end="")
                retrieved = []
            latencies.append((time.time() - t0) * 1000.0)

            mrr_list.append(mrr(retrieved, gold))
            ndcg_list.append(ndcg_at_k(retrieved, gold, 5))
            for k in top_k_vals:
                k_scores[k].append(recall_at_k(retrieved, gold, k))

        elapsed_total = round(time.time() - t_start, 2)
        mean_latency = round(sum(latencies) / len(latencies), 1)

        all_results[name] = {
            "recall": {
                f"@{k}": round(sum(v) / len(v), 4) for k, v in k_scores.items()
            },
            "mrr": round(sum(mrr_list) / len(mrr_list), 4),
            "ndcg@5": round(sum(ndcg_list) / len(ndcg_list), 4),
            "mean_latency_ms": mean_latency,
            "total_time_s": elapsed_total,
        }
        res = all_results[name]
        print(
            f" Done! R@5={res['recall']['@5']:.3f} | MRR={res['mrr']:.3f} | "
            f"nDCG@5={res['ndcg@5']:.3f} | {mean_latency}ms/q"
        )

    # Run Adaptive Router with trace logging
    print("Benchmarking: Adaptive Router (Proposed) ...", end="", flush=True)
    k_scores = {k: [] for k in top_k_vals}
    mrr_list = []
    ndcg_list = []
    latencies = []
    action_counts: Counter[str] = Counter()
    traces: list[dict[str, Any]] = []

    t_start = time.time()
    for item in items:
        q = item["colloquial_query"]
        gold = set(item["gold_chunk_ids"])

        t0 = time.time()
        try:
            retrieved, router_res = run_adaptive_router(q, llm, max_k)
            action_name = router_res.action.value
        except Exception as exc:
            print(f" [Error: {exc}]", end="")
            retrieved = []
            action_name = "error"
            router_res = None

        latencies.append((time.time() - t0) * 1000.0)
        action_counts[action_name] += 1

        rec_5 = recall_at_k(retrieved, gold, 5)
        mrr_val = mrr(retrieved, gold)
        ndcg_val = ndcg_at_k(retrieved, gold, 5)

        mrr_list.append(mrr_val)
        ndcg_list.append(ndcg_val)
        for k in top_k_vals:
            k_scores[k].append(recall_at_k(retrieved, gold, k))

        if router_res:
            traces.append({
                "id": item.get("id"),
                "query": q,
                "gold_chunk_ids": list(gold),
                "retrieved_ids": retrieved,
                "action": action_name,
                "features": router_res.features.to_dict(),
                "recall@5": rec_5,
                "mrr": mrr_val,
            })

    elapsed_total = round(time.time() - t_start, 2)
    mean_latency = round(sum(latencies) / len(latencies), 1)

    action_dist = {
        act: round(cnt / len(items) * 100.0, 1)
        for act, cnt in action_counts.items()
    }

    all_results["Adaptive Router (Proposed)"] = {
        "recall": {
            f"@{k}": round(sum(v) / len(v), 4) for k, v in k_scores.items()
        },
        "mrr": round(sum(mrr_list) / len(mrr_list), 4),
        "ndcg@5": round(sum(ndcg_list) / len(ndcg_list), 4),
        "mean_latency_ms": mean_latency,
        "total_time_s": elapsed_total,
        "action_distribution_pct": action_dist,
    }

    res = all_results["Adaptive Router (Proposed)"]
    print(
        f" Done! R@5={res['recall']['@5']:.3f} | MRR={res['mrr']:.3f} | "
        f"nDCG@5={res['ndcg@5']:.3f} | {mean_latency}ms/q"
    )

    # Save to file
    payload = {
        "summary": all_results,
        "traces": traces,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nDetailed evaluation written to -> {out_path}")
    return all_results


def print_comparison_table(results: dict[str, Any]) -> None:
    """Print clean formatted comparison table."""
    try:
        from tabulate import tabulate

        rows = []
        for name, m in results.items():
            r = m["recall"]
            rows.append([
                name,
                f"{r.get('@1', 0.0):.3f}",
                f"{r.get('@3', 0.0):.3f}",
                f"{r.get('@5', 0.0):.3f}",
                f"{m['mrr']:.3f}",
                f"{m['ndcg@5']:.3f}",
                f"{m['mean_latency_ms']} ms",
            ])

        headers = [
            "Framework Strategy",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "MRR",
            "nDCG@5",
            "Latency/Query",
        ]
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))

        if "Adaptive Router (Proposed)" in results:
            dist = results["Adaptive Router (Proposed)"].get(
                "action_distribution_pct", {}
            )
            print("\nAdaptive Routing Action Breakdown:")
            for act, pct in dist.items():
                print(f"  • {act:22s} : {pct:5.1f}%")
            print()

    except ImportError:
        for name, m in results.items():
            print(f"{name:30s} R@5={m['recall']['@5']:.3f} MRR={m['mrr']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Adaptive Router vs Baselines."
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Run without Ollama LLM"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of eval items"
    )
    parser.add_argument(
        "--out", default=OUT_RESULTS_JSON, help="Output JSON path"
    )
    args = parser.parse_args()

    use_llm = not args.no_llm
    results_dict = evaluate_all(
        llm=use_llm, limit=args.limit, out_path=args.out
    )
    print_comparison_table(results_dict)
