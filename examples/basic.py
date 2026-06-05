"""
Basic Groundy usage — the @groundy decorator, with and without a cache.
Run: python examples/basic.py

groundy makes ONE LLM call of its own — REFORMULATION (rephrasing the question). The
ANSWER call is *yours*: in a real project you already have a `query -> str` function, and
you just put `@groundy` on it — groundy never needs the answer API.

This file is standalone, so it has to build an answer client too (Duty 1 below). That
client + the GROUNDY_ANSWER_* env vars are EXAMPLE SCAFFOLDING ONLY — they exist so the
script has something to answer with. When you drop groundy into your codebase you delete
all of that and decorate your existing call:

    @groundy                       # reformulation auto-configures from OPENAI_* env
    def ask(q): return my_existing_llm_call(q)

Both calls speak the OpenAI API, so each can point at a *different* model and even a
*different provider* (OpenAI, OpenRouter, Groq, Featherless, …) — fully independently,
in code (the kwargs below) or via env (see .env.example).
"""

import os

from dotenv import load_dotenv

load_dotenv()  # load .env so the GROUNDY_* / OPENAI_* defaults are available below

from openai import OpenAI  # noqa: E402

from groundy import groundy, GroundyChecker  # noqa: E402

# Groundy is silent by default. To see reformulations + answers in dev, set
# GROUNDY_DEBUG=1 in your environment (it's in .env.example) — picked up from .env above.

# ======================================================================================
# Duty 1 — ANSWERING (your answer_fn).   ⚠ EXAMPLE SCAFFOLDING ONLY.
# In a real project this already exists — you'd skip this entire block and just decorate
# your existing answer function. groundy never needs the answer API; it only *invokes*
# answer_fn (there is no answer_model= knob on @groundy by design). We build a client here
# purely so this standalone script has something to answer with. Its credentials come from
# the GROUNDY_ANSWER_* / OPENAI_* env vars (see .env.example).
# ======================================================================================
ANSWER_MODEL = os.getenv("GROUNDY_ANSWER_MODEL", "gpt-4o")
answer_client = OpenAI(
    base_url=os.getenv("GROUNDY_ANSWER_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("GROUNDY_ANSWER_API_KEY") or os.getenv("OPENAI_API_KEY"),
)
ANSWER_SYSTEM = "You are a helpful assistant. Answer the user's question."


def call_model(query: str) -> str:
    """The RAW model call. No cache underneath — that's essential (see README)."""
    response = answer_client.chat.completions.create(
        model=ANSWER_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


# ======================================================================================
# Duty 2 — REFORMULATION (groundy's own call).
# Configured IN CODE via @groundy(model=, base_url=, api_key=, …) — a different model and
# even a different provider than the answer client above. Passing literals here works
# identically; we read env first only so the example follows your .env. Leave base_url /
# api_key as None to inherit the generic OPENAI_* credentials.
# ======================================================================================
# base_url / api_key default to None → the library inherits the generic OPENAI_* creds.
# Point them at a different provider (here or via GROUNDY_REFORMULATION_*) to split tasks.
REFORMULATION = dict(
    model=os.getenv("GROUNDY_REFORMULATION_MODEL", "gpt-4o-mini"),
    base_url=os.getenv("GROUNDY_REFORMULATION_BASE_URL"),
    api_key=os.getenv("GROUNDY_REFORMULATION_API_KEY"),
)


# --- 1. The headline API: a decorator that returns a trustworthy string ----------


@groundy(n=5, threshold=0.75, **REFORMULATION)
def ask(query: str) -> str:
    return call_model(query)


# --- 2. Same thing, but with a cache so groundy runs only on a miss --------------
# Any object with get/set works. A plain dict-backed cache is the simplest possible
# one (exact-match keys); swap in Redis / a managed semantic cache in real code.


class DictCache:
    """Toy exact-match cache. A managed semantic cache would match *similar* queries."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: str):
        self._store[key] = value


cache = DictCache()


@groundy(n=5, threshold=0.75, cache=cache, **REFORMULATION)
def ask_cached(query: str) -> str:
    return call_model(query)


if __name__ == "__main__":
    # The decorator returns a plain string — answer or refusal.
    print("\n=== @groundy (string in, string out) ===")
    print(
        ask("Who was the 14th person to walk on the Moon?")
    )  # only 12 exist → refusal

    # With a cache: first call runs the full check, second is an instant hit.
    print("\n=== @groundy(cache=...) — runs only on a miss ===")
    q = "What is the capital of France?"
    print("first  :", ask_cached(q))  # miss → full consistency check
    print("second :", ask_cached(q))  # hit  → returned instantly, no LLM calls

    # Refusals are cached too (the "negative cache"): once a question is judged
    # unreliable, that verdict is stored, so we don't re-interrogate the model next time.
    print("\n=== @groundy(cache=...) — refusals are cached, not re-checked ===")
    q = "Who was the 14th person to walk on the Moon?"
    print("first  :", ask_cached(q))  # miss → full check → refusal, cached
    print("second :", ask_cached(q))  # hit  → same refusal, zero LLM calls

    # The 20% door: full GroundyResult with all the scores. The reformulation duty is
    # configured on the checker (same kwargs as @groundy); the answer duty is answer_fn.
    print("\n=== GroundyChecker.check() — the rich result ===")
    checker = GroundyChecker(n=5, threshold=0.75, **REFORMULATION)
    r = checker.check("What is the capital of France?", answer_fn=call_model)
    print(r)
    print(f"consistency={r.consistency_score:.3f}  best_answer={r.best_answer!r}")
