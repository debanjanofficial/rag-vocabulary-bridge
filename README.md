# rag-vocabulary-bridge

Query-mapping module that bridges vocabulary mismatch between casual user
queries and formal REHAU RAUVISIO product documentation.

```
User query → Mapping module → Retriever → Semantic rerank (top 15) → Answer (Ollama)
```

The LLM Self-Query heuristic guide/spec rerank is unchanged. A separate
cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores candidates for every
strategy, then `qwen3:8b` writes a plain-language grounded answer.

## Project layout

```
rag-vocabulary-bridge-main/
├── app.py                  # Streamlit UI
├── config.py               # paths + model names
├── requirements.txt
├── data/
│   ├── eval_dataset.jsonl  # eval set (formal / colloquial / answer / gold)
│   ├── terminology.json    # term base (concepts + flat_map)
│   └── corpus_chunks.jsonl # indexed knowledge base (build artifact)
├── results/
│   └── evaluation_results.json
├── chroma_db/              # vector index (build artifact)
├── src/                    # runtime library
│   ├── embeddings.py
│   ├── llm.py
│   ├── retriever.py
│   ├── reranker.py         # cross-encoder semantic rerank
│   ├── generator.py        # grounded plain-language answer
│   ├── answer_pipeline.py  # retrieve → rerank → generate
│   ├── qrels.py
│   ├── terminology.py
│   └── strategies/
└── scripts/                # one-time build tools
    ├── build_corpus.py
    ├── build_eval_set.py
    ├── build_terminology.py
    ├── patch_fast_path.py
    ├── index.py
    └── evaluate.py
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/index.py          # if chroma_db not built yet
streamlit run app.py
```

## Rebuild from scratch (slow — needs Ollama)

```bash
python scripts/build_corpus.py
python scripts/build_eval_set.py --target 180
python scripts/build_terminology.py --max-chunks 80
python scripts/patch_fast_path.py
python scripts/index.py
python scripts/evaluate.py --no-llm
```

## Stack

- Embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Mapping LLM: `qwen3:8b` via Ollama (Self-Query)
- Answer LLM: `qwen3:0.6b` via Ollama (fast grounded answers)
- Vector store: ChromaDB
- PDF corpus: external path in `config.py` → `CORPUS_DIR`
- Answer path: retrieve 30 → rerank top 15 (UI) → generate from top 8 (short) or 12 (detailed)
- Answer styles: Short / Detailed — both use `qwen3:0.6b`
  (Detailed = more context + adaptive length; short for facts, longer for how-tos)
- Speed: Ollama uses GPU by default; set `OLLAMA_NUM_GPU=0` for CPU-only.
  Uncheck “Generate answer” in the UI to skip generation.
