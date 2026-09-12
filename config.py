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
GEN_MODEL = "qwen3:8b"          # Self-Query / mapping (needs stronger JSON)
ANSWER_MODEL = "llama3.2:3b"     # Robust plain-language answer generation
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Grounding & NLI (Pillar 5) ───────────────────────────────────────────────
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
NLI_ENTAILMENT_THRESHOLD = 0.35
NLI_CONTRADICTION_THRESHOLD = 0.50
MAX_CANDIDATE_PASSAGES_FOR_NLI = 5

# Ollama GPU layers: unset = let Ollama use GPU; set OLLAMA_NUM_GPU=0 for CPU-only
_OLLAMA_GPU_ENV = os.environ.get("OLLAMA_NUM_GPU")
OLLAMA_NUM_GPU = int(_OLLAMA_GPU_ENV) if _OLLAMA_GPU_ENV is not None else None

QUERY_INSTRUCTION = (
    "Given a user question in everyday language, retrieve REHAU product "
    "manual passages that answer it."
)

# Post-retrieval: wide pool → semantic rerank (display) → generate
RETRIEVE_CANDIDATES = 30
RERANK_TOP_N = 15          # shown in UI after cross-encoder

# Answer length modes (both use ANSWER_MODEL = qwen3:0.6b)
GENERATE_TOP_N = 8         # short mode: chunks into the answer LLM
GENERATE_MAX_CHARS = 400
GENERATE_NUM_PREDICT = 350

GENERATE_TOP_N_DETAILED = 12
GENERATE_MAX_CHARS_DETAILED = 600
GENERATE_NUM_PREDICT_DETAILED = 900

# ── Chunking (used by scripts/build_corpus.py) ──────────────────────────────────
CHUNK_SIZE      = 320
CHUNK_OVERLAP   = 60
MIN_CHUNK_WORDS = 25

# Curated product-line synonyms (input for scripts/patch_fast_path.py)
PRODUCT_LINES = os.path.join(PROJECT_ROOT, "scripts", "product_lines.json")
