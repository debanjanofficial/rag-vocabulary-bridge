"""
Central configuration — paths and model names.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# External PDF corpus (not stored in this repo)
CORPUS_DIR = r"C:\Users\MadMax\Desktop\vocab_missmatch\data\raw\Dataset\RAUVISIO_House_cleaned"

# ── Data (two deliverables + corpus index source) ───────────────────────────────
DATA_DIR         = os.path.join(PROJECT_ROOT, "data")
EVAL_DATASET     = os.path.join(DATA_DIR, "eval_dataset.jsonl")
TERMINOLOGY      = os.path.join(DATA_DIR, "terminology.json")
CORPUS_CHUNKS    = os.path.join(DATA_DIR, "corpus_chunks.jsonl")

# ── Generated outputs ───────────────────────────────────────────────────────────
RESULTS_DIR      = os.path.join(PROJECT_ROOT, "results")
EVAL_RESULTS     = os.path.join(RESULTS_DIR, "evaluation_results.json")
CHROMA_DIR       = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME  = "rehau_corpus"

# ── Models ──────────────────────────────────────────────────────────────────────
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GEN_MODEL   = "qwen3:8b"

QUERY_INSTRUCTION = (
    "Given a user question in everyday language, retrieve REHAU product "
    "manual passages that answer it."
)

# ── Chunking (used by scripts/build_corpus.py) ──────────────────────────────────
CHUNK_SIZE      = 320
CHUNK_OVERLAP   = 60
MIN_CHUNK_WORDS = 25

# Curated product-line synonyms (input for scripts/patch_fast_path.py)
PRODUCT_LINES = os.path.join(PROJECT_ROOT, "scripts", "product_lines.json")
