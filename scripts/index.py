"""Index corpus_chunks.jsonl into ChromaDB with Qwen3-Embedding."""

import json
import os
import shutil

import chromadb

import scripts._bootstrap  # noqa: F401
from config import CORPUS_CHUNKS, CHROMA_DIR, COLLECTION_NAME
from src.embeddings import embed_documents

BATCH = 64


def main():
    if not os.path.exists(CORPUS_CHUNKS):
        raise FileNotFoundError(f"{CORPUS_CHUNKS} not found. Run scripts/build_corpus.py first.")

    with open(CORPUS_CHUNKS, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"Loaded {len(chunks)} chunks")

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metas = [{
            "source_pdf": c["source_pdf"], "product": c["product"],
            "doc_type": c["doc_type"], "page_start": c["page_start"],
            "page_end": c["page_end"],
        } for c in batch]
        collection.add(
            documents=texts,
            embeddings=embed_documents(texts).tolist(),
            ids=ids, metadatas=metas,
        )
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"Done — {len(chunks)} chunks in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
