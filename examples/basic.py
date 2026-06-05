"""
Basic Groundy usage — the @groundy decorator, with and without a cache.
Run: python examples/basic.py   (needs ANTHROPIC_API_KEY)
"""

from dotenv import load_dotenv

load_dotenv()  # load .env so ANTHROPIC_API_KEY / GROUNDY_DEBUG are set before imports

import anthropic  # noqa: E402

from groundy import groundy, GroundyChecker  # noqa: E402

# Groundy is silent by default. To see reformulations + answers in dev, set
# GROUNDY_DEBUG=1 in your environment (it's in .env.example) — picked up from .env above.

client = anthropic.Anthropic()

# Answer however you like — verbose, helpful, your own style. groundy verifies with its
# own terse pass internally (verify_prompt), so you DON'T need to force terseness here;
# the answer you serve stays exactly as written below.
ANSWER_SYSTEM = "You are a helpful assistant. Answer the user's question."


def call_model(query: str) -> str:
    """The RAW model call. No cache underneath — that's essential (see README)."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text


# --- 1. The headline API: a decorator that returns a trustworthy string ----------


@groundy(n=5, threshold=0.75)
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


@groundy(n=5, threshold=0.75, cache=cache)
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

    # The 20% door: full GroundyResult with all the scores.
    print("\n=== GroundyChecker.check() — the rich result ===")
    checker = GroundyChecker(n=5, threshold=0.75)
    r = checker.check("What is the capital of France?", answer_fn=call_model)
    print(r)
    print(f"consistency={r.consistency_score:.3f}  best_answer={r.best_answer!r}")
