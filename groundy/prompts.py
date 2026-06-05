"""
groundy.prompts
~~~~~~~~~~~~~~~
Central home for all LLM prompt text used across the library.

Keeping prompts here (rather than inline in core.py / backends/) makes them
easy to find, tweak, and reuse — e.g. the reformulation prompt and the
llm_judge similarity prompt both live alongside each other.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Query reformulation (used by GroundyChecker._generate_reformulations)
# ----------------------------------------------------------------------

REFORMULATION_SYSTEM = (
    "You are a query reformulation engine. "
    "Given a query, produce exactly the requested number of semantically equivalent "
    "reformulations. Each reformulation must preserve the full meaning but use "
    "different wording, structure, or phrasing. "
    "Return ONLY a JSON array of strings, no other text."
)

REFORMULATION_USER = (
    "Produce {n} semantically equivalent reformulations of this query:\n\n"
    "{query}\n\n"
    "Return a JSON array with exactly {n} strings."
)


# ----------------------------------------------------------------------
# LLM-as-judge similarity (used by backends/llm_judge.py)
# ----------------------------------------------------------------------

JUDGE_PROMPT = """
""".strip()
