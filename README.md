# groundy 🌱

**Keep your LLM grounded - no ground truth required.**

A grounded model agrees with itself: ask the same question a few different ways and the
answer holds. A model that's improvising scatters. `groundy` wraps that check into one
decorator that returns an answer you can trust - or a refusal when the model is just
making things up. No labels, no fine-tuning, no retrieval.

```python
from groundy import groundy

@groundy
def ask(q: str) -> str:
    return my_llm(q)   # your LLM call - any provider, returns a str

ask("Who proved Fermat's Last Theorem?")     # → "Andrew Wiles."
ask("Who was the 14th person on the Moon?")  # → "I'm not confident enough to answer that reliably."
```

Same signature, same `str` return. Nothing downstream changes - the answer just became
trustworthy.

## Get started

**1. Install** (not on PyPI yet):

```bash
uv add git+https://github.com/lopoc/groundy.git
```

That's the full library, ready to use — the `@groundy` decorator and the local `embeddings`
backend work out of the box, no extras needed. Two optional extras add heavier integrations
only if you want them:

| Extra | Adds | Use it for |
|---|---|---|
| `fastembed` | ONNX embedding backend (no torch) | ~15× lighter import (CLI cold start ~10s → ~1–2s). Select with `backend="fastembed"`. |
| `langfuse` | Langfuse tracing adapter | Trace every check (`tracer=LangfuseTracer()`). See [Observability](#observability). |

Add them in the brackets (comma-separated for several) — note the quotes and the `name @`
prefix when you include an extra:

```bash
uv add "groundy[fastembed,langfuse] @ git+https://github.com/lopoc/groundy.git"
```

Skip the extras and nothing breaks: `fastembed` and the Langfuse SDK are imported lazily —
only when you actually select that backend or construct the tracer — so a plain install
never needs them.

**2. Give groundy an API key, a provider, and a model name.** It makes one call of its own
- reformulation, over any OpenAI-compatible API - all under its own `GROUNDY_*` namespace:

```bash
export GROUNDY_API_KEY=sk-...
export GROUNDY_BASE_URL=https://api.openai.com/v1   # your provider — name it, no default (OpenRouter, Groq, a local server…)
export GROUNDY_MODEL=gpt-4o-mini                     # the reformulation model (required, no default)
```

**3. Decorate your LLM call** and use it as usual:

```python
from openai import OpenAI
from groundy import groundy

client = OpenAI()

@groundy
def ask(q: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": q}]
    ).choices[0].message.content

print(ask("Who proved Fermat's Last Theorem?"))
print(ask("Who was the 14th person on the Moon?"))
```

That's it. A ready-to-run version (decorator + cache + raw checker) ships in the repo:
`uv run python examples/basic.py`.

> 💡 `export GROUNDY_DEBUG=1` prints every reformulation, answer, and score.

## Vibe-check it from the terminal

No code needed - `groundy` asks your question a few ways and shows you the *matrix*: each
distinct answer with a bar for how much it **agrees with the rest** (groundy's own signal),
consensus on top, outliers at the bottom. Identical answers collapse to one `×N` row:

```bash
export GROUNDY_API_KEY=sk-...
export GROUNDY_BASE_URL=https://api.openai.com/v1   # your provider — required, no default
export GROUNDY_MODEL=gpt-4o-mini

groundy "Who was the 14th person to walk on the Moon?"
```

```text
🌱 groundy

  ? Who was the 14th person to walk on the Moon?

  ⚠ uncertain   consistency 0.50   · 17.8s

  I'm not confident enough to answer that reliably.

  scatter
    █████░░░ 0.61  Eugene Cernan (the last person to walk on the Moon, Apollo 17)…
    ████░░░░ 0.52  Eugene Cernan was the last (12th) person to walk on the Moon…
    ███░░░░░ 0.41  Harrison Schmitt ×2
```

On a reliable question the bars stand tall together and collapse to one row
(`████████ 1.00  Paris ×5`); on a shaky one they fan down as the answers pull apart.

Want the raw structure? `--matrix` prints the full N×N pairwise heatmap - mutually-agreeing
answers light up as bright blocks, so you *see* the clusters with no threshold and nothing
aggregated:

```text
  scatter
       a b c d e
    a  ██░░██████  Eugene Cernan was the last (12th)…
    b  ░░██░░░░░░  Gene Cernan
    c  ██░░██████  Eugene Cernan was the last (12th)…
    d  ██░░██████  Eugene Cernan was the last (12th)…
    e  ██░░██████  Eugene Cernan (the last person…)
```

It reads `GROUNDY_API_KEY` + `GROUNDY_MODEL` like everything else. Pipe a question in
(`echo "…" | groundy`), add `-q` for answer-only output, `--matrix` for the heatmap,
`-n`/`-t` to tune, or `--debug` for the raw reformulation log.

## How it works

An uncertain model disagrees with itself when you rephrase the question; a confident one
doesn't. With `@groundy(n=5)`, each call:

1. **Rephrases** the query 4 ways - groundy's one own call.
2. **Answers all 5 tersely.** A `verify_prompt` is prepended so the comparison is about
   *substance, not phrasing*. These are the *verify answers*.
3. **Scores agreement** - embeds the verify answers locally (sentence-transformers) and
   averages their pairwise cosine similarity into a `consistency_score` in `[0, 1]`.
4. **Decides:** `reliable = consistency_score >= threshold`.
5. **Answers your way - only if reliable.** It calls your function once more on the raw
   query for the *served* answer (your verbosity/prompt) and returns it. Unreliable → it
   skips this call and returns your `on_unreliable` string.

You serve the answer the way you want it, but verification is terse so verbosity can't
hide disagreement. Cost: **7 LLM calls when reliable** (1 reformulation + 5 verify + 1
served), **6 when unreliable**, all synchronous - which is exactly why you cache it.

## Cache it - pay once per *cluster* of questions

groundy is expensive, so hand it a cache and it runs **only on a miss**. A cache is anything
with `get(key) -> str | None` and `set(key, value)`. The real win is a **semantic** cache: a
hit fires on any question close enough in *meaning*, so groundy runs once per cluster of
similar questions and serves the whole neighbourhood for free.

```python
from groundy import groundy

# Bring any semantic cache exposing get(key) -> str | None and set(key, value). A hit fires
# on questions close in *meaning*, so groundy runs once per cluster (GPTCache, Momento,
# Upstash, Redis + RedisVL - a 3-line adapter if the method names differ).
cache = SemanticCache(threshold=0.9)

@groundy(cache=cache)
def ask(q: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content   # the RAW model

ask("Who discovered penicillin?")          # MISS → full check → verdict cached
ask("Who was penicillin discovered by?")   # HIT  → same meaning, zero LLM calls
```

On a hit groundy never runs. On a miss it checks, then `cache.set`s the verdict - refusals
included, so "the model can't answer this" is remembered too.

> ⚠️ **The one rule: groundy goes *above* your semantic cache, never below it.** If a
> semantic cache sits inside the wrapped call, the reformulations - semantically
> equivalent by design - all hit the same entry, score a perfect 1.0, and *every* check
> falsely passes. The semantic cache belongs on top (via `cache=`), caching the verdict.

## When you want the numbers

The decorator hides the scores on purpose. Reach past it for the rich result:

```python
from groundy import GroundyChecker

checker = GroundyChecker(n=5, threshold=0.75)
r = checker.check("What does Italian Civil Code art. 2043 establish?", answer_fn=my_llm)

r.consistency_score   # 0.0–1.0
r.is_reliable         # bool
r.best_answer         # the served answer if reliable, else None
r.consensus_answer, r.agreement_scores, r.similarity_scores, r.latency_ms
```

`best_answer` is the **served** answer (your raw call) when reliable, and `None` when not
- on a genuine split the right move is to refuse, not guess. The decorator turns that
`None` into your `on_unreliable` string. (`consensus_answer`, the verify answer that agrees
most with the rest, is diagnostic only.)

## Run on any vendor

There are **two independent LLM tasks**, configured separately:

- **Answering** - your decorated function. OpenAI, LiteLLM, Ollama, anything returning a
  `str`. There's no `answer_model=` knob: the answer call *is* your function.
- **Reformulating** - groundy's own OpenAI-compatible call. Set `GROUNDY_MODEL` +
  `GROUNDY_BASE_URL` (both required, no default provider), or pass `model` / `base_url` /
  `api_key`.

So you can reformulate on a cheap, fast model and answer on a stronger one - even across
providers:

```python
@groundy(
    model="llama-3.3-70b-versatile",            # reformulate on Groq…
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
)
def ask(q: str) -> str:
    return openai_client.chat.completions.create(   # …answer on OpenAI
        model="gpt-4o", messages=[{"role": "user", "content": q}]
    ).choices[0].message.content
```

Any OpenAI-compatible endpoint works - that covers OpenAI, OpenRouter, Groq, Together,
Fireworks, and local servers (vLLM, llama.cpp, Ollama).

## Knobs

| Param | Default | What it does |
|---|---|---|
| `n` | `5` | Answers compared: original + n-1 reformulations. Must be ≥ 2. Higher = sturdier + pricier. |
| `threshold` | `0.75` | Score below this → refusal. **Calibrate it** (see limits). |
| `backend` | `"embeddings"` | `embeddings` (local, sentence-transformers) or `llm_judge` (stub). |
| `model` | `None` | Reformulation model - **required** (no default). `None` → `GROUNDY_MODEL`, else `ValueError`. |
| `temperature` | `0.0` | Reformulator temperature (`0.0` = reproducible). Set `None` to omit it for models that reject the param. |
| `base_url` | `None` | Reformulation provider — **required** (no default). `None` → `GROUNDY_BASE_URL`, else `ValueError`. |
| `api_key` | `None` | `None` → `GROUNDY_API_KEY` (may be unset for keyless local servers). |
| `verify_prompt` | *(terse instruction)* | Prepended to the verify answers (not the served one). `None` verifies with your raw answers. |
| `cache` | `None` | Any object with `get`/`set`. Runs groundy only on a miss. |
| `tracer` | `None` | Any object with the `Tracer` protocol. Emits a nested trace per check. Langfuse adapter in `groundy[langfuse]`. |
| `on_unreliable` | *(a refusal)* | Returned/cached when the model disagrees with itself. |

## Honest limits - read this

groundy measures **self-consistency, not correctness.** Know the failure modes:

- **Consistent confabulation passes.** A confidently, consistently wrong model scores
  high. This catches uncertainty *that surfaces as divergence* - a large subset of
  hallucination, not all of it. Terse verify answers help: verbose hedging hides
  disagreement (verbose answers to *"the 14th person on the Moon"* all hedge alike and
  score ~0.9; terse ones confabulate *different* names → ~0.30, flagged). That's why
  verification is terse by default while your served answer stays verbose.
- **Calibrate the threshold.** With the default `all-MiniLM-L6-v2` backend, scores cluster
  high (~0.75–0.95) for any related text. `0.75` is a starting point - tune it on your
  prompts.
- **It costs ~N+2 LLM calls per check** (n=5 ≈ 7, sequential). Hence `cache=`: vet a
  question once, serve it free forever after.

## Observability

Optional and agnostic. Pass a `tracer` (a tiny `Tracer` protocol, just like `cache=`) and
every `check()` emits a nested trace: `reformulate → verify ×n → score → served`. Default
`tracer=None` → no tracing, zero overhead.

A Langfuse adapter ships in the box — add the `langfuse` extra:

```bash
uv add "groundy[langfuse] @ git+https://github.com/lopoc/groundy.git"
```

```python
from groundy.observability.langfuse import LangfuseTracer

@groundy(tracer=LangfuseTracer())   # reads LANGFUSE_* from the env
def ask(q: str) -> str:
    ...
```

The core imports no vendor SDK - only you import the adapter. groundy owns one LLM call
(reformulation), so that node carries the model, temperature, token usage, and a prompt hash;
the `answer_fn` nodes show text + timing only. Prefer to log it yourself? The full
`GroundyResult` is still right there. For dev, `GROUNDY_DEBUG=1` prints reformulations,
answers, and scores.

## Develop

```bash
git clone https://github.com/lopoc/groundy.git
cd groundy
uv sync                              # creates .venv, installs runtime + dev tools

uv run python examples/basic.py      # smoke test (needs GROUNDY_API_KEY + GROUNDY_MODEL)
uv run ruff check groundy            # lint
uv run ruff format groundy           # format
uv run pytest                        # tests (once a tests/ dir exists)
```

## Roadmap

- [x] CLI: `groundy "your query"`
- [ ] `async def acheck()` - parallelize the N calls
- [ ] `llm_judge` backend (structured 0–1 scoring - sharper than embeddings)
- [ ] Tests + benchmark (measured reliable-vs-hallucinated separation)

## Origin

A practical take on the **Laplace agent** from the Socrates/Laplace judicial-AI framework.

MIT License
