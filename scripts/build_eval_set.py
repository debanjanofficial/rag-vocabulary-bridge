"""Generate data/eval_dataset.jsonl from corpus chunks via local LLM."""

import scripts._bootstrap  # noqa: F401
import argparse
import json
import os
import random
import re

from config import CORPUS_CHUNKS, EVAL_DATASET, DATA_DIR
from src.llm import chat_json

SEED = 42

_STOP = {
    "the", "a", "an", "to", "of", "in", "on", "is", "are", "be", "for",
    "with", "that", "this", "and", "or", "not", "it", "as", "at", "by",
    "from", "can", "do", "does", "how", "what", "which", "when", "i",
    "my", "you", "your", "use", "used", "will", "if", "any", "all",
}


def load_chunks() -> list[dict]:
    with open(CORPUS_CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def content_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z][a-z0-9]+", text.lower())
    return {t for t in toks if t not in _STOP and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def alpha_ratio(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    wordish = sum(1 for t in toks if re.fullmatch(r"[A-Za-z][A-Za-z\-]+", t))
    return wordish / len(toks)


def is_usable_chunk(c: dict) -> bool:
    if c["n_words"] < 60:
        return False
    if alpha_ratio(c["text"]) < 0.55:
        return False
    return True


def stratified_sample(chunks: list[dict], max_chunks: int) -> list[dict]:
    rng = random.Random(SEED)

    def doc_rank(dt: str) -> int:
        d = dt.lower()
        if "tech" in d:
            return 0
        if "product" in d:
            return 1
        if d == "root":
            return 2
        return 3

    by_product: dict[str, list[dict]] = {}
    for c in chunks:
        if is_usable_chunk(c):
            by_product.setdefault(c["product"], []).append(c)

    for prod, lst in by_product.items():
        rng.shuffle(lst)
        lst.sort(key=lambda c: doc_rank(c["doc_type"]))

    products = list(by_product.keys())
    rng.shuffle(products)
    picked, cursors = [], {p: 0 for p in products}
    while len(picked) < max_chunks:
        progressed = False
        for p in products:
            i = cursors[p]
            if i < len(by_product[p]):
                picked.append(by_product[p][i])
                cursors[p] += 1
                progressed = True
                if len(picked) >= max_chunks:
                    break
        if not progressed:
            break
    return picked


def already_processed() -> set[str]:
    done = set()
    if os.path.exists(EVAL_DATASET):
        with open(EVAL_DATASET, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["gold_chunk_ids"][0])
                except Exception:
                    pass
    return done


PROMPT_TMPL = """You are creating evaluation data for a RAG system over REHAU \
RAUVISIO product manuals. Work in ENGLISH only.

Read the PASSAGE below. Then produce a JSON object with these keys:
  "answerable": true/false — can a clear, self-contained Q&A be built from THIS passage alone?
  "formal_query": a precise technical question an expert/engineer would ask, answerable only from the passage.
  "answer": a short factual answer (1-3 sentences) using ONLY information in the passage.
  "colloquial_queries": a list of EXACTLY 2 versions of the same question as a NON-EXPERT employee would casually ask it — informal, everyday words, avoid technical jargon and product codes, contractions/typos are fine.

Rules:
- If the passage is just a table of numbers, boilerplate, or has no clear concept, set "answerable": false and leave other fields empty.
- The colloquial queries must NOT reuse the technical terms from the passage (use everyday synonyms instead).
- Output ONLY the JSON object.

PASSAGE:
\"\"\"{chunk}\"\"\"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--max-overlap", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    max_chunks = args.max_chunks or int(args.target * 1.4)

    chunks = load_chunks()
    done = already_processed()
    print(f"Corpus chunks: {len(chunks)} | already processed: {len(done)}")

    sample = [c for c in stratified_sample(chunks, max_chunks)
              if c["chunk_id"] not in done]
    print(f"Will attempt {len(sample)} new chunks (target {args.target})\n")

    accepted = len(done)
    attempted = 0

    fout = open(EVAL_DATASET, "a", encoding="utf-8")
    try:
        for c in sample:
            if accepted >= args.target:
                break
            attempted += 1
            print(f"[{attempted}/{len(sample)}] {c['chunk_id']} ... ", end="", flush=True)

            data = chat_json(PROMPT_TMPL.format(chunk=c["text"]), num_predict=500)
            if not data or not data.get("answerable"):
                print("skip")
                continue

            fq = (data.get("formal_query") or "").strip()
            ans = (data.get("answer") or "").strip()
            colls = [str(x).strip() for x in (data.get("colloquial_queries") or []) if str(x).strip()]
            if not fq or not ans or not colls:
                print("empty, skip")
                continue

            chunk_tok = content_tokens(c["text"])
            scored = sorted(
                ((jaccard(content_tokens(q), chunk_tok), q) for q in colls),
                key=lambda x: x[0],
            )
            kept = [(o, q) for o, q in scored if o <= args.max_overlap]
            if not kept:
                print(f"overlap {scored[0][0]:.2f}, skip")
                continue

            primary_overlap, primary_q = kept[0]
            item = {
                "id": f"q{accepted:04d}",
                "formal_query": fq,
                "colloquial_query": primary_q,
                "colloquial_queries": [q for _, q in kept],
                "answer": ans,
                "gold_chunk_ids": [c["chunk_id"]],
                "product": c["product"],
                "doc_type": c["doc_type"],
                "source_pdf": c["source_pdf"],
                "overlap_score": round(primary_overlap, 3),
                "status": "auto",
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
            accepted += 1
            print(f"OK [{accepted}]")
    finally:
        fout.close()

    print(f"\nDone. {accepted} items -> {EVAL_DATASET}")


if __name__ == "__main__":
    main()
