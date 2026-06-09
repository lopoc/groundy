"""
groundy.backends.embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic similarity via sentence-transformers + cosine similarity.
Runs fully local, no API calls.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the model once and cache it."""
    from sentence_transformers import SentenceTransformer

    # all-MiniLM-L6-v2: fast, small (80MB), good enough for consistency checking
    # swap to 'all-mpnet-base-v2' for better quality at the cost of speed
    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity_batch(texts_a: List[str], texts_b: List[str]) -> List[float]:
    """
    Compute cosine similarity for a list of text pairs.

    Returns one float per pair. Cosine is in [-1, 1]; for related text it sits in
    ~[0, 1], but genuinely opposed answers can score negative — that's intentional
    signal (it drags the consistency score down), so scores are NOT clamped.
    """
    model = _get_model()

    # The caller expands C(n,2) pairs into two aligned lists, so every distinct answer
    # shows up n-1 times across them — encoding all of them is n(n-1) forward passes for
    # only n_distinct unique strings. Embed each distinct string once; the pair scores are
    # then just dot products of cached vectors (the encode is the cost, not the dot).
    uniq = list(dict.fromkeys(texts_a + texts_b))
    vectors = model.encode(uniq, normalize_embeddings=True)
    vec = dict(zip(uniq, vectors))

    # dot product of normalized vectors = cosine similarity (not clamped: opposed answers
    # can score negative, which is intentional signal).
    return [float((vec[a] * vec[b]).sum()) for a, b in zip(texts_a, texts_b)]
