"""Qwen3-Embedding wrapper (asymmetric query vs document encoding)."""

from __future__ import annotations

import numpy as np

from config import EMBED_MODEL, QUERY_INSTRUCTION

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _format_query(text: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"


def embed_documents(texts: list[str], batch_size: int = 32,
                    show_progress: bool = False) -> np.ndarray:
    model = get_model()
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True,
    )


def embed_queries(texts: list[str], batch_size: int = 32,
                  show_progress: bool = False) -> np.ndarray:
    model = get_model()
    formatted = [_format_query(t) for t in texts]
    return model.encode(
        formatted, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True,
    )


def embed_query(text: str) -> list[float]:
    return embed_queries([text])[0].tolist()
