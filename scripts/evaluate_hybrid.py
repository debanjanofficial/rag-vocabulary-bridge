"""Comparative evaluation script for Hybrid Retrieval with Reciprocal Rank Fusion.

Benchmarks:
  1) Dense Only (ChromaDB Cosine Baseline)
  2) BM25 Only (Okapi BM25 Baseline)
  3) Balanced Hybrid RRF (alpha=0.5)
  4) Adaptive Router + Decoupled TKG Hybrid (Proposed)
"""

import argparse
import json
import math
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tabulate import tabulate

from config import DATA_DIR, EVAL_DATASET
from src.retriever import retrieve, retrieve_hybrid, retrieve_routed
from src.router.bm25_index import get_bm25_index
from src.router.router import route_query


def compute_metrics(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k_vals: list[int] = [1, 3, 5],
) -> dict[str, float]:
    """Compute Recall@K, MRR, and nDCG@5."""
    if not expected_chunk_ids:
        return {f"recall@{k}": 0.0 for k in k_vals} | {"mrr": 0.0, "ndcg@5": 0.0}

    retrieved_ids = retrieved_chunk_ids
    metrics = {}

    for k in k_vals:
        top_k = set(retrieved_ids[:k])
        hits = len(top_k.intersection(expected_chunk_ids))
        metrics[f"recall@{k}"] = 1.0 if hits > 0 else 0.0

    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_chunk_ids:
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr

    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:5], start=1):
        if cid in expected_chunk_ids:
            dcg += 1.0 / math.log2(rank + 1)
    metrics["ndcg@5"] = dcg
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Dense vs BM25 vs Hybrid RRF Retrieval."
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Hybrid RRF balance weight (0.0=BM25, 1.0=Dense)"
    )
    args = parser.parse_args()

    with open(EVAL_DATASET, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    if args.limit > 0:
        questions = questions[:args.limit]

    print(f"\n=======================================================")
    print(f"Hybrid Retrieval Benchmark (RRF alpha={args.alpha})")
    print(f"Dataset: {len(questions)} queries")
    print(f"=======================================================")

    bm25_idx = get_bm25_index()

    strategies = {
        "Dense Only (Cosine)": lambda q: retrieve(q, top_k=5),
        "BM25 Only (Okapi)": lambda q: [cid for cid, _ in bm25_idx.search(q, top_k=5)],
        f"Hybrid RRF (alpha={args.alpha})": lambda q: retrieve_hybrid(q, top_k=5, alpha=args.alpha),
        "Adaptive Router + Hybrid RRF (Proposed)": lambda q: retrieve_routed(
            route_query(q, llm_available=False), top_k=5
        ),
    }

    results = {}
    for name, retriever_fn in strategies.items():
        print(f"\nBenchmarking: {name} ...", end="", flush=True)
        t0 = time.perf_counter()
        agg = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0}

        for item in questions:
            q_text = item.get("colloquial_query") or item.get("question", "")
            expected = item.get("gold_chunk_ids") or item.get("ground_truth_chunk_ids") or []
            retrieved_ids = retriever_fn(q_text)

            m = compute_metrics(retrieved_ids, expected)
            for k, v in m.items():
                agg[k] += v

        n = len(questions)
        lat = (time.perf_counter() - t0) * 1000.0 / n
        res = {k: v / n for k, v in agg.items()}
        res["latency_ms"] = lat
        results[name] = res
        print(
            f" Done! R@5={res['recall@5']:.3f} | MRR={res['mrr']:.3f} | nDCG@5={res['ndcg@5']:.3f} | {lat:.1f}ms/q"
        )

    # Comparison Table
    table = []
    for name, r in results.items():
        table.append([
            name,
            f"{r['recall@1']:.3f}",
            f"{r['recall@3']:.3f}",
            f"{r['recall@5']:.3f}",
            f"{r['mrr']:.3f}",
            f"{r['ndcg@5']:.3f}",
            f"{r['latency_ms']:.1f} ms",
        ])

    print("\n" + tabulate(
        table,
        headers=["Retrieval Channel", "Recall@1", "Recall@3", "Recall@5", "MRR", "nDCG@5", "Latency/Query"],
        tablefmt="rounded_grid",
    ))

    out_path = os.path.join(DATA_DIR, "../results/evaluation_results_hybrid.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed evaluation to -> {out_path}")


if __name__ == "__main__":
    main()
