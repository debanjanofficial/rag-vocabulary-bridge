"""Derive ground-truth lookups from eval_dataset.jsonl (no separate qrels file)."""

from __future__ import annotations

import json

from config import EVAL_DATASET


def load_qrels(path: str | None = None) -> dict:
    """
    Map each colloquial query variant to its gold entry:
      {colloquial_query: {id, gold_chunk_ids, formal_query, answer, product}}
    """
    path = path or EVAL_DATASET
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            entry = {
                "id": item["id"],
                "gold_chunk_ids": item["gold_chunk_ids"],
                "formal_query": item["formal_query"],
                "answer": item["answer"],
                "product": item.get("product", ""),
            }
            for q in item.get("colloquial_queries", [item["colloquial_query"]]):
                qrels[q] = entry
    return qrels
