"""
groundy.backends.llm_judge
~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic similarity via LLM-as-judge.

STUB — interface is defined and ready, implementation is TODO.
The signature matches embeddings.py so backends are interchangeable.

When implemented, this will ask Claude to rate semantic equivalence
between answer pairs on a 0-1 scale, which is more robust for
domain-specific or nuanced content but costs extra API calls.
"""

from __future__ import annotations

from typing import List


def judge_similarity_batch(texts_a: List[str], texts_b: List[str]) -> List[float]:
    """
    Rate semantic similarity between pairs of texts using an LLM as judge.

    Parameters
    ----------
    texts_a, texts_b : list of str
        Parallel lists of texts to compare.

    Returns
    -------
    list of float
        Similarity scores in [0, 1].

    TODO:
    - Call Claude with a structured prompt asking for a 0.0-1.0 similarity score
    - Use structured output / tool_use to get a clean float
    - Batch pairs into a single prompt to reduce API calls
    - Cache results for identical pairs
    """
    # When implemented, format groundy.prompts.JUDGE_PROMPT per pair:
    #     JUDGE_PROMPT.format(text_a=a, text_b=b)
    raise NotImplementedError(
        "llm_judge backend is a stub. "
        "Use backend='embeddings' for now, or implement this. "
        "See the docstring for the spec."
    )
