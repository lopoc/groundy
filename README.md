# groundy

**Keep your LLM grounded — no ground truth required.**

A *grounded* model agrees with itself: ask the same question a few different ways and the
answer holds. A model that's improvising scatters. `groundy` wraps that check into a
single decorator that returns an answer you can trust — or a refusal when the model is
just making things up. No labels, no fine-tuning, no retrieval.

```python
from groundy import groundy

@groundy(threshold=0.8)
def ask(q: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content

ask("Who proved Fermat's Last Theorem?")      # → "Andrew Wiles."
ask("Who was the 14th person on the Moon?")   # → "I'm not confident enough to answer that reliably."
```

Same signature, same `str` return type. Nothing downstream changes — the answer just
became trustworthy.

---

## The idea

An uncertain model **disagrees with itself** when you rephrase the question; a
confident one doesn't. groundy exploits that — no labels, no fine-tuning, no service
beyond the model you already call.

```
your query ──▶ rephrase ×(N-1) ──▶ answer each ──▶ pairwise similarity ──▶ score
                                                                              │
                              reliable → consensus answer   ◀─ score ≥ threshold
                              unreliable → refusal string   ◀─ score <  threshold
```

## Get started

**1. Install** — not on PyPI yet, so install straight from GitHub with
[uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/lopoc/groundy.git
```

(Prefer pip? `pip install git+https://github.com/lopoc/groundy.git`. Want to hack on
groundy itself instead? See [Develop](#develop).)

**2. Give groundy a key for its *one* call.** The only LLM call groundy makes on its own is
**reformulation** (rephrasing the question) — so that's all it needs a key for. If your
project already exports `OPENAI_API_KEY`, you're done:

```bash
export OPENAI_API_KEY=sk-...
# optional — run reformulation on any OpenAI-compatible provider (OpenRouter, Groq, …):
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1
# export GROUNDY_REFORMULATION_MODEL=gpt-4o-mini
```

> **groundy wraps your *existing* answer call — it never needs the answer API.** You hand it
> a `query -> str` function (your model, your keys, already wired) and groundy just calls it.
> The `GROUNDY_ANSWER_*` vars in [.env.example](.env.example) /
> [examples/basic.py](examples/basic.py) exist *only* so the standalone example has
> something to answer with — they are **not** part of the library. (Don't even want a key
> for reformulation? Inject `reformulate_fn` and groundy needs no key at all — see *Run on
> any vendor*.)

**3. Wrap any `query -> str` LLM call** with `@groundy` and call it as usual — you get
back a trustworthy answer, or a refusal. Save this as `try_groundy.py` and run it with
`uv run python try_groundy.py`:

```python
from openai import OpenAI
from groundy import groundy

client = OpenAI()   # reads OPENAI_API_KEY / OPENAI_BASE_URL from the env

@groundy(threshold=0.8)
def ask(q: str) -> str:
    # Answer however you like — verbose, your system prompt, your temperature.
    # groundy verifies with its own terse pass; your served answer stays as-is.
    msg = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        messages=[{"role": "user", "content": q}],
    )
    return msg.choices[0].message.content

print(ask("Who proved Fermat's Last Theorem?"))     # → a full answer about Andrew Wiles
print(ask("Who was the 14th person on the Moon?"))  # → "I'm not confident enough to answer that reliably."
```

That's it. The first call makes a handful of LLM calls under the hood (it asks the
question a few ways); the rest of the README is about controlling cost, caching, and
running on other providers. A ready-to-run version (decorator + cache + raw checker)
ships in the repo — `uv run python examples/basic.py`.

> 💡 Want to *see* it work? `export GROUNDY_DEBUG=1` prints every reformulation and
> answer, plus the consistency score and the cache hit/miss for each call.

## Use it

The decorator turns a `query -> str` call into a *trustworthy* `query -> str`. Works
bare or with options:

```python
@groundy
def ask(q: str) -> str:
    return my_llm(q)

@groundy(n=5, threshold=0.8, on_unreliable="I don't know.")
def ask(q: str) -> str:
    return my_llm(q)
```

### What `@groundy` does, step by step

When you call the wrapped function with a query, the decorator runs one full
consistency check before returning. With `@groundy(n=5)`:

1. **Rephrase the query** `n-1` times (here, 4). One reformulation call produces 4
   semantically-equivalent rewordings. This is groundy's **own** call — a *separate* model
   and provider from your answers, set via `GROUNDY_REFORMULATION_*` env / `model=` /
   `base_url=` (or swapped out entirely with `reformulate_fn`; see *Run on any vendor*).
2. **Answer all `n` of them — tersely.** Calls your function on the original query *and*
   each rephrasing, with a terse instruction prepended (`verify_prompt`), so the answers
   compared are short and the signal is about *substance, not phrasing*. These are the
   **verify answers** — *same question, asked 5 ways, answered tersely*.
3. **Score the agreement.** Embeds the 5 verify answers locally (sentence-transformers)
   and averages their pairwise cosine similarity into one `consistency_score` in
   `[0, 1]`. High = the model said the same thing each time; low = it improvised.
4. **Decide.** `reliable = consistency_score >= threshold`.
5. **Answer your way — only if reliable.** *Now* groundy calls your function on the raw
   query for the **served answer**, in your verbosity/temperature/prompt, and returns it.
   If unreliable it **skips this call entirely** and returns your `on_unreliable` string —
   no point spending tokens on an answer you're about to discard.
6. **Return a `str`.** Either way you get a plain string — your answer or the refusal.
   The function's signature never changed; the answer just became trustworthy.

The split matters: you serve the answer *the way you want it*, but the **verification is
terse and deterministic** so verbosity can't hide disagreement. Cost per query:
**7 LLM calls when reliable** (1 reformulation + 5 terse verify + 1 served), **6 when
unreliable** (served answer skipped), all synchronous — which is exactly why you cache it.

## Don't pay for it twice — plug in your cache

groundy is expensive (~N+2 calls per check). You don't want to re-run it for a question
you've already vetted. So hand it your cache and it runs **only on a miss** — vet a
question once, serve it free forever after.

A cache is anything with `get(key) -> str | None` and `set(key, value)` — Redis works
out of the box:

```python
import redis
from groundy import groundy

cache = redis.Redis()   # anything with .get / .set; a semantic cache works too

@groundy(threshold=0.8, cache=cache)   # ← groundy cache.get()s first, cache.set()s on a miss
def ask(q: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content   # the RAW model

ask("Who discovered penicillin?")   # MISS → full check → answer cached
ask("Who discovered penicillin?")   # HIT  → straight from the cache, zero LLM calls
```

The cache check lives **inside** the decorator — that's the point, your code doesn't
change. On every call groundy:

1. `cache.get(q)` — hit? return it. **groundy never runs.**
2. miss → run the consistency check → get a trustworthy answer (or the refusal).
3. `cache.set(q, answer)` — populate, including refusals, so "the model can't answer
   this" is remembered too and short-circuits next time.

**Momento, Upstash, GPTCache** all work the same way (a 3-line adapter if the method
names differ). With a *semantic* cache, a hit fires on *similar* questions too, so
groundy runs **once per cluster of similar questions** and the cache serves the whole
neighbourhood. groundy never stores, embeds, or evicts anything — **retention, TTL and
eviction stay entirely your cache's job.** No `cache=`? The decorator just returns the
string and you wire your own loop.

> ⚠️ **The one rule: groundy goes *above* your semantic cache, never below it.**
> The wrapped call must hit the **raw** model. If a semantic cache sits *inside*
> `answer_fn`, the reformulations — semantically equivalent by design — all hit the same
> cache entry, return identical answers, score a perfect 1.0, and *every* check falsely
> passes. The semantic cache belongs on top (via `cache=`), caching groundy's verdict.

## When you want the numbers

The decorator hides the scores on purpose. Reach past it for the rich result:

```python
from groundy import GroundyChecker

checker = GroundyChecker(n=5, threshold=0.75)
r = checker.check("What does Italian Civil Code art. 2043 establish?", answer_fn=my_llm)

r.consistency_score   # 0.0–1.0
r.is_reliable         # bool
r.best_answer         # the consensus answer if reliable, else None
r.consensus_answer, r.agreement_scores, r.similarity_scores, r.latency_ms, ...
```

`best_answer` returns the **consensus** (the answer that agrees most with the rest) when
reliable, and `None` when not — because on a genuine split the consensus can be a
*popular wrong answer*, so the right move is to refuse, not to guess. The decorator just
turns that `None` into your `on_unreliable` string.

## Run on any vendor

There are **two independent LLM tasks**, and you configure each one separately:

- **Answering** — answers the question *and each reformulation*. This is always yours:
  `answer_fn` (or the decorated function) just returns a `str`. OpenAI, LiteLLM, Ollama, a
  local model, anything.
- **Reformulating** — rephrases the question. This is groundy's **own** OpenAI-compatible
  call. Because the OpenAI API is the de-facto standard, it works as-is against any
  compatible provider — set its `model` / `base_url` / `api_key` directly or via the
  `GROUNDY_REFORMULATION_*` env vars.

Keeping them separate is the point: reformulate on a cheap, fast model and answer on a
stronger one — even on **different providers**. Nothing is hardcoded; resolution is
`kwarg → GROUNDY_REFORMULATION_* → OPENAI_* → built-in default`.

| Provider | `base_url` | Example `model` |
|---|---|---|
| OpenAI | *(default)* | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.1-70b-instruct` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Featherless | `https://api.featherless.ai/v1` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |

```python
from openai import OpenAI
from groundy import groundy

# ANSWER on OpenAI (a strong model)…
answer_client = OpenAI()  # OPENAI_API_KEY

@groundy(
    threshold=0.8,
    # …REFORMULATE on Groq (cheap + fast), a different provider entirely:
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",                            # or GROUNDY_REFORMULATION_API_KEY
)
def ask(q: str) -> str:
    msg = answer_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": q}]
    )
    return msg.choices[0].message.content
```

The reformulation half also configures purely from the environment — set
`GROUNDY_REFORMULATION_*` (falling back to `OPENAI_*`) and the decorator stays a bare
`@groundy`. The answer half is just your function, so it reads whatever *your* code reads;
the `GROUNDY_ANSWER_*` names in [examples/basic.py](examples/basic.py) are a convention of
that example, not something the library consumes.

For a provider that *isn't* OpenAI-compatible (or to run the rephrasings on a totally
different stack), inject your own `reformulate_fn` — a `(query, k) -> list[str]` that
returns `k` semantically-equivalent rephrasings. Pass it and the library touches **no**
vendor SDK at all:

```python
import json
import litellm

def rephrase(query: str, k: int) -> list[str]:
    r = litellm.completion(
        model="ollama/llama3",
        messages=[{"role": "user",
                   "content": f"Return a JSON array of {k} reworded versions of: {query}"}],
    )
    return json.loads(r.choices[0].message.content)

@groundy(threshold=0.8, reformulate_fn=rephrase)   # fully SDK-free
def ask(q: str) -> str:
    return litellm.completion(model="ollama/llama3",
                              messages=[{"role": "user", "content": q}]).choices[0].message.content
```

Reformulation is the quality-sensitive call — a weak model may produce sloppy or
non-equivalent rephrasings, or break the JSON contract — so sanity-check its output if you
swap in a small local model.

## Knobs

| Param | Default | What it does |
|---|---|---|
| `n` | `5` | Answers compared: original + `n-1` reformulations. Must be ≥ 2. Higher = sturdier + pricier. |
| `threshold` | `0.75` | Score below this → refusal. **Calibrate it** (see limits). |
| `backend` | `"embeddings"` | `embeddings` (local, sentence-transformers) or `llm_judge` (stub). |
| `model` | `None` | Reformulation model (groundy's own call — *not* your answer model). `None` → `GROUNDY_REFORMULATION_MODEL`, else `gpt-4o-mini`. Ignored if `reformulate_fn` is set. |
| `temperature` | `0.0` | Default reformulator temperature — `0.0` for reproducible rephrasings. Set `None` to omit it for models that reject the param. Ignored if `reformulate_fn` is set. |
| `base_url` | `None` | OpenAI-compatible endpoint for the reformulator. `None` → `GROUNDY_REFORMULATION_BASE_URL`, then `OPENAI_BASE_URL`, then OpenAI. Set it for OpenRouter, Groq, Featherless, … Ignored if `reformulate_fn` is set. |
| `api_key` | `None` | API key for the reformulator. `None` → `GROUNDY_REFORMULATION_API_KEY`, then `OPENAI_API_KEY`. Lets reformulation use a different key/provider than your answer call. Ignored if `reformulate_fn` is set. |
| `reformulate_fn` | `None` | Bring-your-own reformulator: `(query, k) -> list[str]`. Set it to run on any stack with **no** vendor SDK dependency. |
| `verify_prompt` | *(a terse instruction)* | Prepended to the **verify** answers (not the served one) to force terseness. Set `None` to verify with your raw answers. |
| `cache` | `None` | Any object with `get`/`set`. Drives the get→check→set loop. |
| `on_unreliable` | *(a refusal string)* | Returned/cached when the model disagrees with itself. |

## Honest limits — read this

This measures **self-consistency, not correctness.** Know the failure modes:

- **Consistent confabulation passes.** A model that's *confidently, consistently wrong*
  scores high. This catches uncertainty **that surfaces as divergence** — a large subset
  of hallucination, not all of it. groundy already helps here: the verify answers are
  forced terse (`verify_prompt`), because verbosity hides disagreement. Verbose answers
  to *"the 14th person on the Moon"* (only 12 exist) all hedge alike and score ~0.9
  (hallucination **hidden**); terse ones confabulate *different* names → ~0.30, flagged.
  This is why verification is terse by default while your served answer stays verbose.
- **Calibrate the threshold.** With the default `all-MiniLM-L6-v2` backend, scores
  cluster high (~0.75–0.95) for any related text. `0.75` is a starting point — tune it
  on your own prompts.
- **Costs ~N+1 LLM calls per check** (n=5 ≈ 6, sequential, ~15s). This is exactly why
  the `cache=` integration exists: vet a question once, serve it free forever after.

## Observability

There's no built-in vendor integration — by design, groundy does one thing. You have the
full `GroundyResult`, so log it however you already do:

```python
r = checker.check(query, answer_fn)
my_tracer.log(consistency=r.consistency_score, reliable=r.is_reliable)
```

For dev, set `GROUNDY_DEBUG=1` to see reformulations + answers (silent otherwise).

## Develop

Want to hack on groundy itself? Clone and sync with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/lopoc/groundy.git
cd groundy
uv sync                              # creates .venv, installs runtime + dev tools

uv run python examples/basic.py      # smoke test (needs OPENAI_API_KEY)
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

A practical take on the **Laplace agent** from the Socrates/Laplace judicial-AI
framework

MIT License
