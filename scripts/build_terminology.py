"""Build data/terminology.json from corpus chunks via local LLM."""

import scripts._bootstrap  # noqa: F401
import argparse
import json
import os
import random
import re

from config import CORPUS_CHUNKS, EVAL_DATASET, TERMINOLOGY
from src.llm import chat_json

RAW_PATH = os.path.join(os.path.dirname(__file__), "_term_raw.jsonl")
SEED = 42


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_chunks() -> list[dict]:
    with open(CORPUS_CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def alpha_ratio(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if re.fullmatch(r"[A-Za-z][A-Za-z\-]+", t)) / len(toks)


def select_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
    rng = random.Random(SEED)

    def rank(dt: str) -> int:
        d = dt.lower()
        if "tech" in d:
            return 0
        if "product" in d:
            return 1
        if d == "root":
            return 2
        return 3

    usable = [c for c in chunks if c["n_words"] >= 60 and alpha_ratio(c["text"]) >= 0.55]
    by_product: dict[str, list[dict]] = {}
    for c in usable:
        by_product.setdefault(c["product"], []).append(c)
    for lst in by_product.values():
        rng.shuffle(lst)
        lst.sort(key=lambda c: rank(c["doc_type"]))

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


PROMPT_TMPL = """You are a terminology engineer building a controlled vocabulary \
for REHAU RAUVISIO products. Work in ENGLISH only.

From the PASSAGE, extract the domain CONCEPTS it defines or describes. For each
concept return an object with:
  "preferred_term": the standardized technical term (controlled vocabulary).
  "acronyms": list of acronyms/abbreviations for it (may be empty).
  "synonyms_colloquial": list of everyday / non-expert ways a layperson might
      refer to this concept (plain words, NOT the technical term).
  "definition": one-sentence definition grounded in the passage.
  "characteristics": list of 1-3 distinguishing features/attributes.

Return JSON: {{"concepts": [ ... ]}}. If no clear concept, return {{"concepts": []}}.
Output ONLY the JSON object.

PASSAGE:
\"\"\"{chunk}\"\"\"
"""


def processed_ids() -> set[str]:
    done = set()
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["chunk_id"])
                except Exception:
                    pass
    return done


def extract(max_chunks: int):
    chunks = load_chunks()
    done = processed_ids()
    sample = [c for c in select_chunks(chunks, max_chunks) if c["chunk_id"] not in done]
    print(f"Extracting concepts from {len(sample)} chunks (already done: {len(done)})")

    with open(RAW_PATH, "a", encoding="utf-8") as fout:
        for i, c in enumerate(sample, 1):
            print(f"[{i}/{len(sample)}] {c['chunk_id']} ... ", end="", flush=True)
            data = chat_json(PROMPT_TMPL.format(chunk=c["text"]), num_predict=600)
            concepts = (data or {}).get("concepts", []) if data else []
            fout.write(json.dumps({"chunk_id": c["chunk_id"], "concepts": concepts},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{len(concepts)} concepts")


def eval_colloquial_phrases() -> set[str]:
    phrases = set()
    if os.path.exists(EVAL_DATASET):
        with open(EVAL_DATASET, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for q in item.get("colloquial_queries", [item.get("colloquial_query", "")]):
                    phrases.add(q.lower().strip())
    return phrases


def merge():
    if not os.path.exists(RAW_PATH):
        print("No raw extractions found. Run without --merge-only first.")
        return

    merged: dict[str, dict] = {}
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for con in rec.get("concepts", []):
                pt = (con.get("preferred_term") or "").strip()
                if not pt:
                    continue
                key = pt.lower()
                m = merged.setdefault(key, {
                    "concept_id": "c_" + slugify(pt),
                    "preferred_term": pt,
                    "acronyms": set(),
                    "synonyms_colloquial": set(),
                    "definition": "",
                    "characteristics": set(),
                })
                for a in con.get("acronyms", []) or []:
                    if a.strip():
                        m["acronyms"].add(a.strip())
                for s in con.get("synonyms_colloquial", []) or []:
                    if s.strip():
                        m["synonyms_colloquial"].add(s.strip())
                for ch in con.get("characteristics", []) or []:
                    if ch.strip():
                        m["characteristics"].add(ch.strip())
                if not m["definition"] and (con.get("definition") or "").strip():
                    m["definition"] = con["definition"].strip()

    concepts = []
    for m in merged.values():
        concepts.append({
            "concept_id": m["concept_id"],
            "preferred_term": m["preferred_term"],
            "acronyms": sorted(m["acronyms"]),
            "synonyms_colloquial": sorted(m["synonyms_colloquial"]),
            "definition": m["definition"],
            "characteristics": sorted(m["characteristics"]),
        })
    concepts.sort(key=lambda c: c["preferred_term"].lower())

    eval_phrases = eval_colloquial_phrases()
    flat: dict[str, str] = {}
    dropped = 0
    for c in concepts:
        for syn in c["synonyms_colloquial"] + c["acronyms"]:
            s = syn.lower().strip()
            if not s or s == c["preferred_term"].lower():
                continue
            if any(s in ep for ep in eval_phrases):
                dropped += 1
                continue
            flat.setdefault(s, c["preferred_term"])

    with open(TERMINOLOGY, "w", encoding="utf-8") as f:
        json.dump({"concepts": concepts, "flat_map": flat}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(concepts)} concepts, {len(flat)} flat mappings -> {TERMINOLOGY}")
    print(f"({dropped} flat entries dropped for eval leakage)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=80)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if not args.merge_only:
        extract(args.max_chunks)
    merge()


if __name__ == "__main__":
    main()
