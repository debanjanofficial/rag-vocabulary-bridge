"""Alpha sensitivity sweep for RRF hybrid fusion on 188 evaluation queries.

Produces publication-ready data showing how reciprocal rank fusion balance weight
(alpha: 0.0 = pure BM25, 1.0 = pure Dense) affects Recall@1, Recall@5, and MRR.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from tabulate import tabulate

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import EVAL_DATASET, RESULTS_DIR
import src.embeddings as emb_module
from src.retriever import retrieve_hybrid

_EMBED_CACHE: dict[str, np.ndarray] = {}
_ORIG_EMBED = emb_module.embed_queries


def _cached_embed(texts: list[str], batch_size: int = 32, show_progress: bool = False):
    uncached = [t for t in texts if t not in _EMBED_CACHE]
    if uncached:
        embs = _ORIG_EMBED(uncached, batch_size=batch_size, show_progress=show_progress)
        for t, emb in zip(uncached, embs):
            _EMBED_CACHE[t] = emb
    return np.array([_EMBED_CACHE[t] for t in texts])


emb_module.embed_queries = _cached_embed


def main():
    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    all_queries = [s.get("colloquial_query") or s.get("question", "") for s in samples]
    print(f"Pre-caching {len(all_queries)} queries...")
    t0 = time.perf_counter()
    _cached_embed(all_queries, batch_size=32)
    print(f"Cached in {time.perf_counter() - t0:.2f}s.")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    sweep_results = []

    print("\nStarting Alpha Sensitivity Analysis:")
    for alpha in alphas:
        r1_list, r5_list, mrr_list, ndcg_list = [], [], [], []
        for s in samples:
            q = s.get("colloquial_query") or s.get("question", "")
            gold = set(s.get("gold_chunk_ids") or s.get("ground_truth_chunk_ids") or [])
            ret = retrieve_hybrid(q, top_k=5, alpha=alpha)

            r1_list.append(1.0 if any(c in gold for c in ret[:1]) else 0.0)
            r5_list.append(1.0 if any(c in gold for c in ret[:5]) else 0.0)

            rr = 0.0
            for rank, c in enumerate(ret[:5], 1):
                if c in gold:
                    rr = 1.0 / rank
                    break
            mrr_list.append(rr)

            dcg = 0.0
            for rank, c in enumerate(ret[:5], 1):
                if c in gold:
                    dcg += 1.0 / math.log2(rank + 1)
            ndcg_list.append(dcg)

        res = {
            "alpha": alpha,
            "recall@1": round(float(np.mean(r1_list)), 4),
            "recall@5": round(float(np.mean(r5_list)), 4),
            "mrr": round(float(np.mean(mrr_list)), 4),
            "ndcg@5": round(float(np.mean(ndcg_list)), 4),
        }
        sweep_results.append(res)
        print(f"  alpha={alpha:.2f} | R@1={res['recall@1']:.3f} | R@5={res['recall@5']:.3f} | MRR={res['mrr']:.3f}")

    table = [[r["alpha"], r["recall@1"], r["recall@5"], r["mrr"], r["ndcg@5"]] for r in sweep_results]
    print("\n" + tabulate(table, headers=["Alpha", "Recall@1", "Recall@5", "MRR", "nDCG@5"], tablefmt="rounded_grid"))

    out_path = os.path.join(RESULTS_DIR, "alpha_sensitivity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved alpha sensitivity curve -> {out_path}")


if __name__ == "__main__":
    main()
