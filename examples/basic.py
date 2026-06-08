"""
Basic groundy usage — the @groundy decorator, with and without a cache.
Run: uv run python examples/basic.py   (needs GROUNDY_API_KEY + GROUNDY_MODEL)

groundy makes ONE LLM call of its own: REFORMULATION (rephrasing the question), over any
OpenAI-compatible API — it needs just three things, all under its own namespace: an API key,
a provider, and a model name (GROUNDY_API_KEY / GROUNDY_BASE_URL / GROUNDY_MODEL). The ANSWER
call is *yours* — the function you decorate. In a real project that function already exists
and you just put `@groundy` on it. This example answers on the same provider and the same
model it reformulates with, so GROUNDY_API_KEY + GROUNDY_MODEL runs the whole thing.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # load .env so GROUNDY_* are available below

from openai import OpenAI  # noqa: E402

from groundy import groundy, GroundyChecker  # noqa: E402

# Your answer call. Reuses the same provider AND model groundy reformulates with, so a
# single GROUNDY_API_KEY + GROUNDY_MODEL runs the whole thing on one model.
client = OpenAI(base_url=os.getenv("GROUNDY_BASE_URL"), api_key=os.getenv("GROUNDY_API_KEY"))
ANSWER_MODEL = os.environ["GROUNDY_MODEL"]


def call_model(query: str) -> str:
    """The RAW model call. No cache underneath — that's essential (see README)."""
    msg = client.chat.completions.create(
        model=ANSWER_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": query}],
    )
    return msg.choices[0].message.content


# --- 1. The headline API: a decorator that returns a trustworthy string ----------


@groundy
def ask(query: str) -> str:
    return call_model(query)


# --- 2. Same thing, but with a cache so groundy runs only on a miss --------------
# Any object with get/set works. A dict is the simplest possible cache (exact-match
# keys); swap in Redis / a managed semantic cache in real code.


class DictCache:
    """Toy exact-match cache. A managed semantic cache would match *similar* queries."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: str):
        self._store[key] = value


cache = DictCache()


@groundy(cache=cache)
def ask_cached(query: str) -> str:
    return call_model(query)


if __name__ == "__main__":
    # The decorator returns a plain string — answer or refusal.
    print("\n=== @groundy (string in, string out) ===")
    print(ask("Who was the 14th person to walk on the Moon?"))  # only 12 exist → refusal

    # With a cache: first call runs the full check, second is an instant hit.
    print("\n=== @groundy(cache=...) — runs only on a miss ===")
    q = "What is the capital of France?"
    print("first  :", ask_cached(q))  # miss → full consistency check
    print("second :", ask_cached(q))  # hit  → returned instantly, no LLM calls

    # Refusals are cached too: once a question is judged unreliable, that verdict is
    # stored, so we don't re-interrogate the model next time.
    print("\n=== @groundy(cache=...) — refusals are cached, not re-checked ===")
    q = "Who was the 14th person to walk on the Moon?"
    print("first  :", ask_cached(q))  # miss → full check → refusal, cached
    print("second :", ask_cached(q))  # hit  → same refusal, zero LLM calls

    # The 20% door: full GroundyResult with all the scores.
    print("\n=== GroundyChecker.check() — the rich result ===")
    checker = GroundyChecker()
    r = checker.check("What is the capital of France?", answer_fn=call_model)
    print(r)
    print(f"consistency={r.consistency_score:.3f}  best_answer={r.best_answer!r}")
