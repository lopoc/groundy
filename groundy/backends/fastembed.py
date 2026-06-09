"""
groundy.backends.fastembed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic similarity via fastembed (ONNX Runtime) + cosine similarity.

Same model as the default ``embeddings`` backend — ``all-MiniLM-L6-v2`` — but run through
ONNX Runtime instead of torch/sentence-transformers, so the import is ~15x lighter (~0.7s
vs ~4.8s) and there's no torch in the process. Opt-in: ``backend="fastembed"`` (needs the
``fastembed`` extra). Embedding quality is identical; only the engine differs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the ONNX model once and cache it (first call downloads the model)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL_NAME)


def cosine_similarity_batch(texts_a: List[str], texts_b: List[str]) -> List[float]:
    """
    Compute cosine similarity for a list of text pairs (fastembed/ONNX engine).

    Mirrors the default ``embeddings`` backend: each distinct string is embedded once
    (the caller hands in the expanded pair lists, so every answer repeats n-1 times), then
    the pair scores are dot products of the cached, L2-normalised vectors. Cosine is NOT
    clamped — opposed answers can score negative, which is intentional signal.
    """
    model = _get_model()

    uniq = list(dict.fromkeys(texts_a + texts_b))
    if not uniq:
        return []
    # fastembed.embed yields one vector per input; normalise to unit length so dot = cosine
    # (don't assume the engine normalises for us — divide explicitly, guarding zero norm).
    vec = {}
    for text, v in zip(uniq, model.embed(uniq)):
        v = np.asarray(v, dtype=np.float64)
        norm = np.linalg.norm(v)
        vec[text] = v / norm if norm else v

    return [float(np.dot(vec[a], vec[b])) for a, b in zip(texts_a, texts_b)]
