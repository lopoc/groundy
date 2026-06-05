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
    all_texts = texts_a + texts_b
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    n = len(texts_a)
    emb_a = embeddings[:n]
    emb_b = embeddings[n:]

    # dot product of normalized vectors = cosine similarity
    scores = (emb_a * emb_b).sum(axis=1).tolist()
    return [float(s) for s in scores]
