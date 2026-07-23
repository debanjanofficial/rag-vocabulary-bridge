"""Evaluate mapping strategies against eval_dataset.jsonl."""

import argparse
import json
import math
import os
import time

from config import EVAL_DATASET, EVAL_RESULTS, GEN_MODEL
from src.retriever import retrieve, retrieve_rrf, retrieve_self_query
from src.strategies.nlp_strategy import nlp_map
from src.strategies.embedding_strategy import embedding_map
from src.strategies.llm_strategy import llm_map


def recall_at_k(retrieved, gold, k):
    return 1.0 if any(r in gold for r in retrieved[:k]) else 0.0


def mrr(retrieved, gold):
    for i, rid in enumerate(retrieved):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, gold, k):
    for i, rid in enumerate(retrieved[:k]):
        if rid in gold:
            return 1.0 / math.log2(i + 2)
    return 0.0


def run_baseline(q, _llm, k):
    return retrieve(q, k)


def run_nlp(q, _llm, k):
    mapped = nlp_map(q)
    queries = mapped.get("retrieval_queries") or [q]
    return retrieve_rrf(queries, top_k=k)


def run_embedding(q, _llm, k):
    mapped = embedding_map(q)
    queries = mapped.get("retrieval_queries") or [mapped.get("final_query", q)]
    return retrieve_rrf(queries, top_k=k)


def run_llm_self_query(q, llm, k):
    mapped = llm_map(q, llm_available=llm)
    filters = mapped.get("filters") or {}
    return retrieve_self_query(
        mapped.get("original_query") or q,
        mapped.get("semantic_query") or mapped.get("final_query") or q,
        product=filters.get("product"),
        doc_type=filters.get("doc_type"),
        prefer_source=mapped.get("prefer_source"),
        top_k=k,
    )


def evaluate(llm=True, top_k_vals=(1, 3, 5), limit=0, out_path=EVAL_RESULTS):
    with open(EVAL_DATASET, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if limit:
        items = items[:limit]
    MAX_K = max(top_k_vals)
    print(f"{len(items)} eval items\n")

    strategies = {"Baseline": run_baseline, "NLP": run_nlp, "Embedding": run_embedding}
    if llm:
        strategies["LLM Self-Query"] = run_llm_self_query
    else:
        print("LLM strategies skipped (--no-llm).\n")

    all_results = {}
    for name, fn in strategies.items():
        print(f"Running: {name} ...")
        k_scores = {k: [] for k in top_k_vals}
        mrr_list, ndcg_list = [], []
        t0 = time.time()
        for item in items:
            q, gold = item["colloquial_query"], set(item["gold_chunk_ids"])
            try:
                retrieved = fn(q, llm, MAX_K)
            except Exception as e:
                print(f"    error: {e}")
                retrieved = []
            mrr_list.append(mrr(retrieved, gold))
            ndcg_list.append(ndcg_at_k(retrieved, gold, 5))
            for k in top_k_vals:
                k_scores[k].append(recall_at_k(retrieved, gold, k))
        elapsed = round(time.time() - t0, 1)
        all_results[name] = {
            "recall": {f"@{k}": round(sum(v) / len(v), 4) for k, v in k_scores.items()},
            "mrr": round(sum(mrr_list) / len(mrr_list), 4),
            "ndcg@5": round(sum(ndcg_list) / len(ndcg_list), 4),
            "time_s": elapsed,
        }
        r = all_results[name]
        print(f"  R@5={r['recall']['@5']:.3f}  MRR={r['mrr']:.3f}  nDCG@5={r['ndcg@5']:.3f}  [{elapsed}s]")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults -> {out_path}")
    return all_results


def print_table(results):
    try:
        from tabulate import tabulate
        rows = [[
            n, m["recall"].get("@1", "-"), m["recall"].get("@3", "-"),
            m["recall"].get("@5", "-"), m["mrr"], m["ndcg@5"], f"{m['time_s']}s"
        ] for n, m in results.items()]
        print(tabulate(rows, headers=["Strategy", "R@1", "R@3", "R@5", "MRR", "nDCG@5", "Time"],
                       tablefmt="rounded_outline", floatfmt=".3f"))
    except ImportError:
        for n, m in results.items():
            print(f"{n:20} R@5={m['recall']['@5']:.3f}  MRR={m['mrr']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=EVAL_RESULTS)
    args = ap.parse_args()

    use_llm = not args.no_llm
    if use_llm:
        try:
            import ollama
            ollama.show(GEN_MODEL)
            print(f"Ollama: {GEN_MODEL}\n")
        except Exception:
            print("Ollama not available — retrieval-only.\n")
            use_llm = False

    print_table(evaluate(llm=use_llm, limit=args.limit, out_path=args.out))
