"""Extract and chunk PDFs into data/corpus_chunks.jsonl."""

import scripts._bootstrap  # noqa: F401
import json
import os
import re

import pdfplumber

from config import (
    CORPUS_DIR, CORPUS_CHUNKS, DATA_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS,
)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def doc_type_abbrev(doc_type: str) -> str:
    d = doc_type.lower()
    if "tech" in d:
        return "tech"
    if "market" in d:
        return "mkt"
    if "product" in d:
        return "prod"
    if d == "root":
        return "root"
    return "gen"


def parse_metadata(pdf_path: str) -> tuple[str, str]:
    rel = os.path.relpath(pdf_path, CORPUS_DIR)
    parts = rel.split(os.sep)
    if len(parts) == 1:
        return "General", "Root"
    if len(parts) == 2:
        return parts[0], "General"
    return parts[0], parts[1]


def collect_pdfs(root_dir: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".pdf"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def extract_words_with_pages(pdf_path: str) -> list[tuple[str, int]]:
    words = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for w in text.split():
                words.append((w, page_no))
    return words


def chunk_words(word_pages: list[tuple[str, int]]) -> list[dict]:
    chunks, step = [], max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    start, n = 0, len(word_pages)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        window = word_pages[start:end]
        if len(window) >= MIN_CHUNK_WORDS:
            chunks.append({
                "text": " ".join(w for w, _ in window).strip(),
                "page_start": window[0][1],
                "page_end": window[-1][1],
                "n_words": len(window),
            })
        if end == n:
            break
        start += step
    return chunks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    pdfs = collect_pdfs(CORPUS_DIR)
    print(f"Found {len(pdfs)} PDFs under {CORPUS_DIR}")

    all_chunks, per_product_idx, failed = [], {}, []
    for pdf_path in pdfs:
        product, doc_type = parse_metadata(pdf_path)
        rel = os.path.relpath(pdf_path, CORPUS_DIR)
        try:
            word_pages = extract_words_with_pages(pdf_path)
        except Exception as e:
            failed.append((rel, str(e)))
            continue
        if len(word_pages) < MIN_CHUNK_WORDS:
            failed.append((rel, "too little text"))
            continue
        p_slug, d_abbr = slugify(product), doc_type_abbrev(doc_type)
        for c in chunk_words(word_pages):
            idx = per_product_idx.get(p_slug, 0)
            per_product_idx[p_slug] = idx + 1
            all_chunks.append({
                "chunk_id": f"{p_slug}_{d_abbr}_{idx:04d}",
                "text": c["text"],
                "source_pdf": rel.replace(os.sep, "/"),
                "product": product,
                "doc_type": doc_type,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "n_words": c["n_words"],
            })

    with open(CORPUS_CHUNKS, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks -> {CORPUS_CHUNKS}")
    if failed:
        print(f"Skipped {len(failed)} PDFs")


if __name__ == "__main__":
    main()
