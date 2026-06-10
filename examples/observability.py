"""
Observability — tracing a groundy check to Langfuse.
Run: uv run --extra langfuse python examples/observability.py

Needs the usual GROUNDY_* vars (the reformulation + answer calls) PLUS Langfuse creds, which
the Langfuse SDK reads itself — groundy core never touches them:
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL

Observability is agnostic: groundy emits a nested trace to any object matching the `Tracer`
protocol. Here we use the shipped Langfuse adapter (`groundy[langfuse]` extra). Pass it via
`tracer=` exactly like `cache=` — `tracer=None` (the default) runs untraced, zero overhead.

Each check shows up in Langfuse as one trace:
    groundy.check
    ├─ reformulate   (the ONE call groundy owns — carries token usage)
    ├─ verify ×n     (your answer_fn, terse — text + timing only, no tokens*)
    ├─ score         (pairwise similarity → consistency verdict)
    └─ served        (your answer_fn on the raw query — only when reliable)
* groundy only sees a string out of answer_fn, never its model or token counts.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # load .env so GROUNDY_* and LANGFUSE_* are available below

from openai import OpenAI  # noqa: E402

from groundy import GroundyChecker, groundy  # noqa: E402
from groundy.observability.langfuse import LangfuseTracer  # noqa: E402

# Your answer call — the same single-model setup as examples/basic.py.
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


# One tracer, reused across calls. It builds a Langfuse client from the LANGFUSE_* env vars.
# flush_on_end pushes each trace before this short script exits (the default).
tracer = LangfuseTracer()


# --- 1. The decorator, now traced ------------------------------------------------
# `tracer=` slots in beside the usual options, just like `cache=`.


@groundy(tracer=tracer)
def ask(query: str) -> str:
    return call_model(query)


if __name__ == "__main__":
    # A reliable question and an unreliable one — two distinct traces in Langfuse.
    print("\n=== @groundy(tracer=...) — every check is a trace ===")
    print("reliable   :", ask("What is the capital of France?"))
    print("unreliable :", ask("Who was the 14th person to walk on the Moon?"))  # only 12 → refusal

    # The rich door is traced the same way — pass tracer= to GroundyChecker.
    print("\n=== GroundyChecker.check(tracer=...) — rich result + trace ===")
    checker = GroundyChecker(tracer=tracer)
    r = checker.check("What is the capital of France?", answer_fn=call_model)
    print(
        f"consistency={r.consistency_score:.3f}  reliable={r.is_reliable}  best={r.best_answer!r}"
    )

    print("\nOpen your Langfuse project to see the groundy.check traces.")
