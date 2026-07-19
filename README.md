# rag-vocabulary-bridge

Query-mapping module that bridges vocabulary mismatch between casual user
queries and formal REHAU RAUVISIO product documentation.

```
User query → Mapping module → Optimized query → Retriever (Qwen3-Embedding)
```

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
- LLM (local): `qwen3:8b` via Ollama
- Vector store: ChromaDB
- PDF corpus: external path in `config.py` → `CORPUS_DIR`
