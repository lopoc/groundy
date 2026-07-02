# groundy 🌱

**Keep your LLM grounded — no ground truth required.**

A grounded model agrees with itself: ask the same question a few different ways and the
answer holds. A model that's improvising scatters. `groundy` wraps that check into one
decorator that returns an answer you can trust — or a refusal when the model is just making
things up. No labels, no fine-tuning, no retrieval.

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

## Install

Not on PyPI yet:

```bash
uv add git+https://github.com/lopoc/groundy.git
```

That's the whole library — the `@groundy` decorator and the local `embeddings` backend work
out of the box. Two optional extras, lazily imported (skip them and nothing breaks):

| Extra | Adds | For |
|---|---|---|
| `fastembed` | ONNX embedding backend (no torch) | ~15× lighter import. Select with `backend="fastembed"`. |
| `langfuse` | Langfuse tracing adapter | Trace every check. See [Observability](#observability). |

```bash
uv add "groundy[fastembed,langfuse] @ git+https://github.com/lopoc/groundy.git"
```

## Use it

groundy makes **one** LLM call of its own — reformulation — over any OpenAI-compatible API.
Point it at a provider, all under the `GROUNDY_*` namespace:

```bash
export GROUNDY_API_KEY=sk-...
export GROUNDY_BASE_URL=https://api.openai.com/v1   # your provider — required, no default
export GROUNDY_MODEL=gpt-4o-mini                     # reformulation model — required, no default
```

Then put `@groundy` on the LLM call you already have:

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
```

That's the whole thing. A runnable version (decorator + cache + raw checker) ships in the
repo: `uv run python examples/basic.py`.

> 💡 `export GROUNDY_DEBUG=1` prints every reformulation, answer, and score.

## Or vibe-check from the terminal

```bash
groundy "Who was the 14th person to walk on the Moon?"
```

groundy asks your question a few ways and shows each distinct answer with a bar for how much
it **agrees with the rest** (its own signal) — consensus on top, outliers at the bottom.
Identical answers collapse to one `×N` row:

```text
🌱 groundy

  ? Who was the 14th person to walk on the Moon?

  ⚠ uncertain   consistency 0.50   · 17.8s

  I'm not confident enough to answer that reliably.

    █████░░░ 0.61  Eugene Cernan (the last person to walk on the Moon, Apollo 17)…
    ████░░░░ 0.52  Eugene Cernan was the last (12th) person to walk on the Moon…
    ███░░░░░ 0.41  Harrison Schmitt ×2
```

A reliable question collapses to one tall row (`████████ 1.00  Paris ×5`); a shaky one fans
down as the answers pull apart. Add `--matrix` for the raw N×N pairwise heatmap, `-q` for
answer-only, `-n`/`-t` to tune, `--debug` for the log. Pipe a question in with
`echo "…" | groundy`.

## How it works

With `@groundy(n=5)`, each call:

1. **Rephrases** the query 4 ways — groundy's one own call.
2. **Answers all 5 tersely.** A `verify_prompt` forces brevity, so the comparison is about
   *substance, not phrasing*. These are the *verify answers*.
3. **Scores agreement** — embeds the verify answers locally (sentence-transformers) and
   averages their pairwise cosine similarity into a `consistency_score` in `[0, 1]`.
4. **Decides:** `reliable = consistency_score >= threshold`.
5. **Answers your way — only if reliable.** One more call on the raw query, returned exactly
   as you wrote it. Unreliable → it skips this call and returns your `on_unreliable` string.

You serve the answer however you like, but verification stays terse so verbosity can't hide
disagreement. Cost: **7 LLM calls when reliable** (1 reformulation + 5 verify + 1 served),
**6 when unreliable** — which is exactly why you cache it.

## Cache it — pay once per *cluster* of questions

Hand groundy a cache and it runs **only on a miss**. A cache is anything with
`get(key) -> str | None` and `set(key, value)`. The real win is a **semantic** cache: a hit
fires on any question close in *meaning*, so groundy runs once per cluster of similar
questions and serves the whole neighbourhood for free.

```python
@groundy(cache=cache)              # GPTCache, Momento, Upstash, Redis+RedisVL… or a dict
def ask(q: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content   # the RAW model

ask("Who discovered penicillin?")          # MISS → full check → verdict cached
ask("Who was penicillin discovered by?")   # HIT  → same meaning, zero LLM calls
```

Refusals are cached too, so "the model can't answer this" is remembered.

> ⚠️ **The one rule: groundy goes *above* your semantic cache, never below it.** A semantic
> cache *inside* the wrapped call collapses the reformulations — equivalent by design — to one
> entry, scores a perfect 1.0, and *every* check falsely passes. Put it on top via `cache=`.

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

`best_answer` is the **served** answer (your raw call) when reliable, `None` when not — on a
genuine split the right move is to refuse, not guess. The decorator turns that `None` into
your `on_unreliable` string. (`consensus_answer`, the verify answer that agrees most with the
rest, is diagnostic only.)

## Run on any vendor

Two **independent** LLM tasks, configured separately:

- **Answering** — your decorated function. OpenAI, LiteLLM, Ollama, anything returning a
  `str`. There's no `answer_model=` knob: the answer call *is* your function.
- **Reformulating** — groundy's own OpenAI-compatible call. Set `GROUNDY_MODEL` +
  `GROUNDY_BASE_URL` (both required), or pass `model` / `base_url` / `api_key`.

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

Any OpenAI-compatible endpoint works — OpenAI, OpenRouter, Groq, Together, Fireworks, and
local servers (vLLM, llama.cpp, Ollama).

## Knobs

| Param | Default | What it does |
|---|---|---|
| `n` | `5` | Answers compared: original + n-1 reformulations. Must be ≥ 2. Higher = sturdier + pricier. |
| `threshold` | `0.75` | Score below this → refusal. **Calibrate it** (see limits). |
| `backend` | `"embeddings"` | `embeddings` (sentence-transformers, local), `fastembed` (ONNX, lighter import), or `llm_judge` (stub). |
| `model` | `None` | Reformulation model — **required**. `None` → `GROUNDY_MODEL`, else `ValueError`. |
| `temperature` | `0.0` | Reformulator temperature (`0.0` = reproducible). `None` omits it for models that reject the param. |
| `base_url` | `None` | Reformulation provider — **required**. `None` → `GROUNDY_BASE_URL`, else `ValueError`. |
| `api_key` | `None` | `None` → `GROUNDY_API_KEY` (may be unset for keyless local servers). |
| `verify_prompt` | *(terse instruction)* | Prepended to the verify answers, not the served one. `None` verifies with your raw answers. |
| `concurrency` | `2` | Verify answers fetched in parallel (`1` = sequential); the served call stays sequential. |
| `cache` | `None` | Any object with `get`/`set`. Runs groundy only on a miss. |
| `tracer` | `None` | Any object with the `Tracer` protocol. Emits a nested trace per check. Langfuse adapter in `groundy[langfuse]`. |
| `on_unreliable` | *(a refusal)* | Returned/cached when the model disagrees with itself. |

## Honest limits — read this

groundy measures **self-consistency, not correctness.** Know the failure modes:

- **Consistent confabulation passes.** A confidently, consistently wrong model scores high.
  This catches uncertainty *that surfaces as divergence* — a large subset of hallucination,
  not all of it. Terse verify answers help: verbose hedging looks alike (~0.9), terse answers
  confabulate *different* names (~0.30, flagged). That's why verification is terse by default
  while your served answer stays verbose.
- **Calibrate the threshold.** With the default `all-MiniLM-L6-v2` backend, scores cluster
  high (~0.75–0.95) for any related text. `0.75` is a starting point — tune it on your prompts.
- **It costs ~N+2 LLM calls per check** (n=5 ≈ 7, sequential). Hence `cache=`.

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

The core imports no vendor SDK — only you import the adapter. groundy owns one call
(reformulation), so that node carries the model, temperature, and token usage; the
`answer_fn` nodes show text + timing only.

## Develop

```bash
git clone https://github.com/lopoc/groundy.git && cd groundy
uv sync                              # .venv + runtime + dev tools
uv run python examples/basic.py      # smoke test (needs GROUNDY_API_KEY + GROUNDY_MODEL)
uv run ruff check groundy            # lint
uv run ruff format groundy           # format
uv run pytest                        # tests
```

## Roadmap

- [x] CLI: `groundy "your query"`
- [ ] `async def acheck()` — parallelize the N calls
- [ ] `llm_judge` backend (structured 0–1 scoring — sharper than embeddings)
- [ ] Tests + benchmark (measured reliable-vs-hallucinated separation)

## Origin

A practical take on the **Laplace agent** from the Socrates/Laplace judicial-AI framework.

MIT License
