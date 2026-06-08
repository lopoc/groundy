# groundy

**Keep your LLM grounded — no ground truth required.**

A grounded model agrees with itself: ask the same question a few different ways and the
answer holds. A model that's improvising scatters. `groundy` wraps that check into one
decorator that returns an answer you can trust — or a refusal when the model is just
making things up. No labels, no fine-tuning, no retrieval.

```python
from groundy import groundy

@groundy
def ask(q: str) -> str:
    return my_llm(q)   # your LLM call — any provider, returns a str

ask("Who proved Fermat's Last Theorem?")     # → "Andrew Wiles."
ask("Who was the 14th person on the Moon?")  # → "I'm not confident enough to answer that reliably."
```

Same signature, same `str` return. Nothing downstream changes — the answer just became
trustworthy.

## Get started

**1. Install** (not on PyPI yet):

```bash
uv add git+https://github.com/lopoc/groundy.git
```

**2. Give groundy an API key, a provider, and a model name.** It makes one call of its own
— reformulation — read like any OpenAI client:

```bash
export OPENAI_API_KEY=sk-...
export GROUNDY_MODEL=gpt-4o-mini           # the reformulation model (required, no default)
# optional — point at any OpenAI-compatible provider (defaults to OpenAI):
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1
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

## How it works

An uncertain model disagrees with itself when you rephrase the question; a confident one
doesn't. With `@groundy(n=5)`, each call:

1. **Rephrases** the query 4 ways — groundy's one own call.
2. **Answers all 5 tersely.** A `verify_prompt` is prepended so the comparison is about
   *substance, not phrasing*. These are the *verify answers*.
3. **Scores agreement** — embeds the verify answers locally (sentence-transformers) and
   averages their pairwise cosine similarity into a `consistency_score` in `[0, 1]`.
4. **Decides:** `reliable = consistency_score >= threshold`.
5. **Answers your way — only if reliable.** It calls your function once more on the raw
   query for the *served* answer (your verbosity/prompt) and returns it. Unreliable → it
   skips this call and returns your `on_unreliable` string.

You serve the answer the way you want it, but verification is terse so verbosity can't
hide disagreement. Cost: **7 LLM calls when reliable** (1 reformulation + 5 verify + 1
served), **6 when unreliable**, all synchronous — which is exactly why you cache it.

## Cache it — pay once per question

groundy is expensive, so hand it a cache and it runs **only on a miss**. A cache is
anything with `get(key) -> str | None` and `set(key, value)` — Redis works out of the box:

```python
import redis
cache = redis.Redis()

@groundy(cache=cache)
def ask(q: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content   # the RAW model

ask("Who discovered penicillin?")   # MISS → full check → answer cached
ask("Who discovered penicillin?")   # HIT  → straight from the cache, zero LLM calls
```

On a hit groundy never runs. On a miss it checks, then `cache.set`s the verdict — refusals
included, so "the model can't answer this" is remembered too. Momento, Upstash, GPTCache
all work the same way; with a *semantic* cache a hit fires on *similar* questions, so
groundy runs once per cluster.

> ⚠️ **The one rule: groundy goes *above* your semantic cache, never below it.** If a
> semantic cache sits inside the wrapped call, the reformulations — semantically
> equivalent by design — all hit the same entry, score a perfect 1.0, and *every* check
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
— on a genuine split the right move is to refuse, not guess. The decorator turns that
`None` into your `on_unreliable` string. (`consensus_answer`, the verify answer that agrees
most with the rest, is diagnostic only.)

## Run on any vendor

There are **two independent LLM tasks**, configured separately:

- **Answering** — your decorated function. OpenAI, LiteLLM, Ollama, anything returning a
  `str`. There's no `answer_model=` knob: the answer call *is* your function.
- **Reformulating** — groundy's own OpenAI-compatible call. Reads `OPENAI_*` by default;
  point it elsewhere with `model` / `base_url` / `api_key`.

So you can reformulate on a cheap, fast model and answer on a stronger one — even across
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

| Provider | `base_url` | Example `model` |
|---|---|---|
| OpenAI | *(default)* | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.1-70b-instruct` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Featherless | `https://api.featherless.ai/v1` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |

For a provider that *isn't* OpenAI-compatible, inject your own reformulator — a
`(query, k) -> list[str]` returning `k` rephrasings — and the library touches **no** vendor
SDK:

```python
@groundy(reformulate_fn=my_rephraser)
def ask(q: str) -> str: ...
```

## Knobs

| Param | Default | What it does |
|---|---|---|
| `n` | `5` | Answers compared: original + n-1 reformulations. Must be ≥ 2. Higher = sturdier + pricier. |
| `threshold` | `0.75` | Score below this → refusal. **Calibrate it** (see limits). |
| `backend` | `"embeddings"` | `embeddings` (local, sentence-transformers) or `llm_judge` (stub). |
| `model` | `None` | Reformulation model — **required** (no default). `None` → `GROUNDY_MODEL`, else `ValueError`. Ignored if `reformulate_fn` is set. |
| `temperature` | `0.0` | Reformulator temperature (`0.0` = reproducible). Set `None` to omit it for models that reject the param. |
| `base_url` / `api_key` | `None` | Reformulation provider. `None` → `OPENAI_BASE_URL` / `OPENAI_API_KEY`. |
| `reformulate_fn` | `None` | Bring-your-own reformulator `(query, k) -> list[str]`; no vendor SDK. |
| `verify_prompt` | *(terse instruction)* | Prepended to the verify answers (not the served one). `None` verifies with your raw answers. |
| `cache` | `None` | Any object with `get`/`set`. Runs groundy only on a miss. |
| `on_unreliable` | *(a refusal)* | Returned/cached when the model disagrees with itself. |

## Honest limits — read this

groundy measures **self-consistency, not correctness.** Know the failure modes:

- **Consistent confabulation passes.** A confidently, consistently wrong model scores
  high. This catches uncertainty *that surfaces as divergence* — a large subset of
  hallucination, not all of it. Terse verify answers help: verbose hedging hides
  disagreement (verbose answers to *"the 14th person on the Moon"* all hedge alike and
  score ~0.9; terse ones confabulate *different* names → ~0.30, flagged). That's why
  verification is terse by default while your served answer stays verbose.
- **Calibrate the threshold.** With the default `all-MiniLM-L6-v2` backend, scores cluster
  high (~0.75–0.95) for any related text. `0.75` is a starting point — tune it on your
  prompts.
- **It costs ~N+2 LLM calls per check** (n=5 ≈ 7, sequential). Hence `cache=`: vet a
  question once, serve it free forever after.

## Observability

None built in — by design. You have the full `GroundyResult`; log it however you already do
(`my_tracer.log(consistency=r.consistency_score, reliable=r.is_reliable)`). For dev, set
`GROUNDY_DEBUG=1` to print reformulations, answers, and scores.

## Develop

```bash
git clone https://github.com/lopoc/groundy.git
cd groundy
uv sync                              # creates .venv, installs runtime + dev tools

uv run python examples/basic.py      # smoke test (needs OPENAI_API_KEY + GROUNDY_MODEL)
uv run ruff check groundy            # lint
uv run black groundy                 # format
uv run pytest                        # tests (once a tests/ dir exists)
```

## Roadmap

- [ ] `llm_judge` backend (structured 0–1 scoring — sharper than embeddings)
- [ ] `async def acheck()` — parallelize the N calls
- [ ] Tests + benchmark (measured reliable-vs-hallucinated separation)
- [ ] CLI: `groundy-check "your query"`

## Origin

A practical take on the **Laplace agent** from the Socrates/Laplace judicial-AI framework.

MIT License
