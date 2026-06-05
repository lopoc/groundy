"""
groundy.core
~~~~~~~~~~~~
Hallucination detection via semantic consistency checking.

The idea (from the Socrates/Laplace judicial AI framework):
  1. Take the original query
  2. Generate N semantically equivalent reformulations
  3. Ask the LLM each reformulation independently
  4. Measure pairwise semantic similarity across all answers
  5. Low consistency → the model is uncertain → flag as potential hallucination

No ground truth needed. The headline API is the ``@groundy`` decorator: it turns a
``query -> str`` LLM call into a *trustworthy* ``query -> str`` — same signature,
same return type, but the answer is either one you can trust or a refusal string.
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from openai import OpenAI
from loguru import logger

from groundy.prompts import REFORMULATION_SYSTEM, REFORMULATION_USER

# --- Reformulation provider config -------------------------------------------------
# Reformulation is the library's OWN LLM call — kept deliberately separate from the call
# you answer with (answer_fn), so the two tasks can run on different models AND providers
# (e.g. reformulate on a cheap/fast model, answer on a stronger one). Nothing is
# hardcoded: model, base_url and api_key all resolve from the environment. Resolution
# order, highest priority first:
#     explicit kwarg  →  GROUNDY_REFORMULATION_*  →  OPENAI_*  →  built-in fallback
# base_url=None means "OpenAI proper"; api_key=None lets the OpenAI SDK find OPENAI_API_KEY.
FALLBACK_REFORMULATION_MODEL = "gpt-4o-mini"


def _env_first(*names: str) -> Optional[str]:
    """Return the first non-empty value among the named environment variables, else None."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# Deterministic by default: the reformulation call uses temperature 0 so the rephrasings
# are stable and reproducible across runs. Set temperature=None to omit the parameter
# (required for models that reject it, e.g. some reasoning models).
TEMPERATURE = 0.0

# Returned (and cached) in place of an answer when the model disagrees with itself.
# Override per-decorator with on_unreliable=.
DEFAULT_REFUSAL = "I'm not confident enough to answer that reliably."

# Prepended to every query when producing the *verify* answers — the ones compared for
# consistency. Forcing terse answers makes divergence about substance, not phrasing or
# length, so the consistency signal is clean. The answer actually *served* is produced
# separately, exactly as the dev wrote answer_fn (their verbosity, their temperature).
# Set verify_prompt=None to disable and verify with the dev's raw answers instead.
VERIFY_PROMPT = (
    "Answer as briefly as possible — just the answer itself, "
    "with no explanation, no caveats, and no preamble."
)


@runtime_checkable
class Cache(Protocol):
    """Minimal cache interface ``@groundy(cache=...)`` drives.

    Any object with these two methods works — a dict, Redis, a managed semantic
    cache (Momento, Upstash, GPTCache), etc. groundy never stores, embeds, or
    evicts anything itself: retention/TTL/eviction stay entirely the cache's job.
    """

    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str) -> None: ...


@dataclass
class GroundyResult:
    original_query: str
    original_answer: str  # the SERVED answer (dev's raw answer_fn); "" when unreliable — only produced if reliable
    reformulations: list[str]
    answers: list[str]  # the terse VERIFY answers, aligned to [query, *reformulations]
    similarity_scores: list[float]  # pairwise, over the verify answers
    consistency_score: float  # mean of pairwise similarities, 0-1
    is_reliable: bool  # consistency_score >= threshold
    threshold: float
    backend: str
    latency_ms: float
    # Consensus = the verify answer that agrees most with the others (the medoid).
    # Diagnostic only — the served answer (original_answer) is what best_answer returns.
    consensus_answer: str = ""  # most-representative verify answer
    consensus_index: int = 0  # its index into answers
    agreement_scores: list[float] = field(  # per-answer mean similarity to the others,
        default_factory=list  # aligned with answers
    )
    metadata: dict = field(default_factory=dict)

    @property
    def all_answers(self) -> list[str]:
        """The verify set: the terse answers compared for consistency."""
        return list(self.answers)

    @property
    def best_answer(self) -> Optional[str]:
        """The answer to serve, or None if the check is unreliable.

        Returns the **served** answer (``original_answer`` — the dev's own answer to the
        raw query, in their style) only when ``is_reliable``. When unreliable, returns
        None: the model disagrees with itself across rephrasings, so the right move is to
        refuse/escalate rather than serve a possibly-confabulated answer. The terse verify
        answers are what *decided* reliability; this is what you actually hand back.
        """
        return self.original_answer if self.is_reliable else None

    @property
    def answer(self) -> Optional[str]:
        """Alias of :attr:`best_answer` — the answer to trust, or None if unreliable."""
        return self.best_answer

    def __repr__(self):
        status = "✓ RELIABLE" if self.is_reliable else "⚠ UNCERTAIN"
        return (
            f"GroundyResult({status} | "
            f"consistency={self.consistency_score:.3f} | "
            f"threshold={self.threshold} | "
            f"n={len(self.reformulations) + 1})"
        )


class GroundyChecker:
    """
    Core checker. Use directly for the rich :class:`GroundyResult`, or reach for the
    ``@groundy`` decorator for the simple ``str``-in/``str``-out path.

    Parameters
    ----------
    n : int
        Total number of answers in the consistency set: the original query's
        answer plus (n-1) reformulation answers. So n=5 means original + 4
        reformulations. Must be >= 2. More = more reliable, more expensive;
        3–5 is a good range for production.
    threshold : float
        Minimum consistency score to consider the answer reliable. 0.75 is a
        reasonable default; lower for exploratory use, higher for critical paths.
    backend : str
        'embeddings' (default) or 'llm_judge' (stub, ready for later).
    model : str | None
        OpenAI-compatible model for the *default* reformulator — the reformulation task
        only; your answers use whatever model ``answer_fn`` calls, independently. None
        (default) reads ``GROUNDY_REFORMULATION_MODEL``, else falls back to ``gpt-4o-mini``.
        Ignored if you pass ``reformulate_fn``.
    temperature : float | None
        Temperature for the *default* reformulator. Defaults to 0.0 for reproducible
        rephrasings. Set to None to omit the parameter — required for models that
        reject it (e.g. some reasoning models). Ignored if you pass ``reformulate_fn``.
    base_url : str | None
        Base URL for the *default* reformulator's OpenAI-compatible endpoint. None (default)
        reads ``GROUNDY_REFORMULATION_BASE_URL`` then ``OPENAI_BASE_URL``, else hits OpenAI.
        Set it (here or via env) to reformulate on OpenRouter, Groq, Featherless, a local
        server, … — a *different provider* than your answer call. Ignored if you pass
        ``reformulate_fn``.
    api_key : str | None
        API key for the *default* reformulator. None (default) reads
        ``GROUNDY_REFORMULATION_API_KEY`` then ``OPENAI_API_KEY``. Lets the reformulation
        provider authenticate with a different key than your answer call. Ignored if you
        pass ``reformulate_fn``.
    reformulate_fn : Callable[[str, int], list[str]] | None
        Bring your own reformulator and the library touches **no** vendor SDK. It takes
        ``(query, k)`` and returns ``k`` semantically-equivalent rephrasings. When None
        (default), an OpenAI-compatible reformulator is used (needs OPENAI_API_KEY).
        Use this to run on any vendor (LiteLLM, a local model, …).
    verify_prompt : str | None
        Prepended to each query when producing the *verify* answers (those compared for
        consistency), to force terse answers and a clean signal. The *served* answer is
        produced separately from the dev's raw ``answer_fn``, untouched. Set to None to
        verify with the dev's raw answers instead.
    """

    def __init__(
        self,
        n: int = 5,
        threshold: float = 0.75,
        backend: str = "embeddings",
        model: Optional[str] = None,
        temperature: Optional[float] = TEMPERATURE,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reformulate_fn: Optional[Callable[[str, int], list[str]]] = None,
        verify_prompt: Optional[str] = VERIFY_PROMPT,
    ):
        if n < 2:
            raise ValueError(
                f"n must be >= 2 (need at least 2 answers to compare), got {n}."
            )
        self.n = n
        self.threshold = threshold
        self.backend = backend
        # Reformulation provider config — resolved kwarg → GROUNDY_REFORMULATION_* env →
        # OPENAI_* env → fallback. Kept entirely separate from the answer call (answer_fn),
        # so the two tasks can run on different models and providers.
        self.model = (
            model
            or _env_first("GROUNDY_REFORMULATION_MODEL")
            or FALLBACK_REFORMULATION_MODEL
        )
        self.temperature = temperature
        self.base_url = base_url or _env_first(
            "GROUNDY_REFORMULATION_BASE_URL", "OPENAI_BASE_URL"
        )
        self.api_key = api_key or _env_first(
            "GROUNDY_REFORMULATION_API_KEY", "OPENAI_API_KEY"
        )
        self.reformulate_fn = reformulate_fn
        self.verify_prompt = verify_prompt

        # lazy-load the similarity backend
        self._similarity_fn = self._load_backend(backend)

        # OpenAI-compatible client for the default reformulator, constructed on first use
        # only (so callers who pass reformulate_fn never need OPENAI_API_KEY).
        self._client: Optional[OpenAI] = None

    def _load_backend(self, backend: str) -> Callable:
        if backend == "embeddings":
            from groundy.backends.embeddings import cosine_similarity_batch

            return cosine_similarity_batch
        elif backend == "llm_judge":
            from groundy.backends.llm_judge import judge_similarity_batch

            return judge_similarity_batch
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. Use 'embeddings' or 'llm_judge'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, query: str, answer_fn: Callable[[str], str]) -> GroundyResult:
        """
        Run the full consistency check and return a rich :class:`GroundyResult`.

        Parameters
        ----------
        query : str
            The original query sent to the LLM.
        answer_fn : Callable[[str], str]
            A function that takes a query string and returns an answer string. This is
            the actual LLM call you're wrapping. **It must hit the raw model** — if a
            *semantic* cache sits underneath it, the reformulations (which are
            semantically equivalent by design) all collapse to the same cached answer,
            similarity is trivially 1.0, and every check falsely passes.

        Two answering modes use this one ``answer_fn``: the **verify** answers — those
        compared for consistency — are produced first, from the query and each
        reformulation with ``verify_prompt`` prepended (terse, clean signal). The
        **served** answer (the dev's raw, verbose call) is produced *only if the verdict
        is reliable* — when unreliable we'd discard it for the refusal, so it's skipped to
        save tokens. So an unreliable check costs one fewer ``answer_fn`` call.
        """
        t0 = time.monotonic()
        logger.debug(
            "check start | n={} threshold={} backend={}",
            self.n,
            self.threshold,
            self.backend,
        )
        logger.debug("[query] {!r}", query)

        # Step 1: generate N-1 reformulations of the original query — via the injected
        # reformulator if given (vendor-agnostic), else the default OpenAI one.
        k = self.n - 1
        if self.reformulate_fn is not None:
            reformulations = list(self.reformulate_fn(query, k))
        else:
            reformulations = self._generate_reformulations(query)

        # Step 2: produce the VERIFY answers — terse, so consistency is about substance,
        # not phrasing/length. Prepend verify_prompt to the original query *and* every
        # reformulation, then answer each with the same model. These N answers (not the
        # verbose served one) are what get compared.
        verify_inputs = [query] + reformulations
        answers = []
        for i, q in enumerate(verify_inputs, start=1):
            vq = f"{self.verify_prompt}\n\n{q}" if self.verify_prompt else q
            a = answer_fn(vq)
            answers.append(a)
            logger.debug("[verify {}/{}] query: {!r}", i, len(verify_inputs), q)
            logger.debug("[verify {}/{}] answer: {!r}", i, len(verify_inputs), a)

        # Step 3: pairwise similarity across the N verify answers
        index_pairs = list(itertools.combinations(range(len(answers)), 2))
        similarity_scores = self._similarity_fn(
            [answers[i] for i, _ in index_pairs],
            [answers[j] for _, j in index_pairs],
        )

        # Step 4: compute consistency score (mean of all pairwise similarities)
        consistency_score = (
            sum(similarity_scores) / len(similarity_scores)
            if similarity_scores
            else 0.0
        )
        is_reliable = consistency_score >= self.threshold

        # Step 5: pick the consensus verify answer — the one agreeing most with the others
        # (the medoid). Diagnostic only: the served answer (below) is what we return.
        agreement_scores = self._agreement_scores(
            len(answers), index_pairs, similarity_scores
        )
        consensus_index = (
            max(range(len(answers)), key=agreement_scores.__getitem__) if answers else 0
        )
        consensus_answer = answers[consensus_index] if answers else ""

        logger.debug(
            "result | consistency={:.3f} reliable={} ({} pairwise scores)",
            consistency_score,
            is_reliable,
            len(similarity_scores),
        )
        logger.debug(
            "consensus | index={} agreement={:.3f} answer={!r}",
            consensus_index,
            agreement_scores[consensus_index] if agreement_scores else 0.0,
            consensus_answer,
        )

        # Step 6: only NOW produce the served answer — the dev's verbose call. Skip it
        # entirely when unreliable: we'd discard it for the refusal anyway, so this saves
        # a (often max_tokens-heavy) generation on every flagged query.
        if is_reliable:
            original_answer = answer_fn(query)
            logger.debug("[served] answer: {!r}", original_answer)
        else:
            original_answer = ""
            logger.debug("[served] skipped — unreliable, no served answer produced")

        latency_ms = (time.monotonic() - t0) * 1000

        return GroundyResult(
            original_query=query,
            original_answer=original_answer,
            reformulations=reformulations,
            answers=answers,
            similarity_scores=similarity_scores,
            consistency_score=consistency_score,
            is_reliable=is_reliable,
            threshold=self.threshold,
            backend=self.backend,
            latency_ms=latency_ms,
            consensus_answer=consensus_answer,
            consensus_index=consensus_index,
            agreement_scores=agreement_scores,
        )

    @staticmethod
    def _agreement_scores(n, index_pairs, similarity_scores):
        """Per-answer mean similarity to the other n-1 answers, indexed by answer position.

        The answer with the highest agreement is the consensus (medoid): an outlier
        answer ends up with a low score, so consensus shifts to the agreeing majority.
        """
        if n <= 1:
            return [0.0] * n
        sums = [0.0] * n
        for (i, j), s in zip(index_pairs, similarity_scores):
            sums[i] += s
            sums[j] += s
        return [sums[k] / (n - 1) for k in range(n)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_reformulations(self, query: str) -> list[str]:
        """Ask the model to rephrase the query into n-1 semantically equivalent variants.

        The original query's answer makes up the n-th member of the consistency set.
        """
        n_reformulations = self.n - 1
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        client = self._client

        kwargs = dict(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": REFORMULATION_SYSTEM},
                {
                    "role": "user",
                    "content": REFORMULATION_USER.format(
                        n=n_reformulations, query=query
                    ),
                },
            ],
        )
        # Pin temperature for reproducibility; omit it entirely when None (some models,
        # e.g. certain reasoning models, reject the parameter).
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = client.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())


# ------------------------------------------------------------------
# The decorator — the headline API
# ------------------------------------------------------------------


def groundy(
    fn: Optional[Callable[..., str]] = None,
    *,
    n: int = 5,
    threshold: float = 0.75,
    backend: str = "embeddings",
    model: Optional[str] = None,
    temperature: Optional[float] = TEMPERATURE,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    reformulate_fn: Optional[Callable[[str, int], list[str]]] = None,
    verify_prompt: Optional[str] = VERIFY_PROMPT,
    cache: Optional[Cache] = None,
    on_unreliable: str = DEFAULT_REFUSAL,
):
    """
    Turn a ``query -> str`` LLM call into a *trustworthy* ``query -> str``.

    The wrapped function keeps its signature and still returns a ``str`` — but the
    string is either an answer the model agrees with itself on, or ``on_unreliable``
    when it doesn't. Downstream code never has to change.

    Works bare or called::

        @groundy
        def ask(q: str) -> str: ...

        @groundy(threshold=0.8)
        def ask(q: str) -> str: ...

    Pass ``cache=`` (any object with ``get``/``set``) to short-circuit on a hit and
    populate on a miss — groundy runs *only when not cached*::

        @groundy(cache=my_semantic_cache)
        def ask(q: str) -> str:
            return client.chat.completions.create(...).choices[0].message.content   # RAW model

    IMPORTANT: the wrapped call must hit the **raw** model. If a *semantic* cache sits
    underneath it, the reformulations collapse to one cached answer and every check
    falsely passes. The semantic cache belongs on top (via ``cache=``), not inside.

    Two independent LLM tasks, configured independently — each can use a different model
    and even a different provider:

    * **Answering** (the question *and* each reformulation) is ``func`` itself. There is
      deliberately no ``answer_model=`` knob: the answer call *is* the function you
      decorate, so you configure its model/provider/key/system-prompt in code, right there
      in ``func`` (use whatever SDK you like — OpenAI, LiteLLM, Ollama, a local model).
    * **Reformulating** (rephrasing the query) is groundy's own OpenAI-compatible call,
      configured here via the ``model`` / ``base_url`` / ``api_key`` / ``temperature``
      kwargs — **in code** — or the matching ``GROUNDY_REFORMULATION_*`` env vars (which
      fall back to ``OPENAI_*``). Explicit kwargs win over env; nothing is hardcoded.

    So you can reformulate on a cheap/fast model and answer on a stronger one, even across
    providers (OpenAI, OpenRouter, Groq, Featherless, …). For a fully custom reformulator on
    any stack, pass ``reformulate_fn=(query, k) -> list[str]`` and the library touches
    **no** vendor SDK::

        @groundy(reformulate_fn=my_rephraser)
        def ask(q: str) -> str: ...

    Served vs. verified: the answer you get back is produced from the raw query exactly
    as you wrote ``func`` (your verbosity, temperature, prompt). The answers groundy
    *compares* for consistency are produced separately with ``verify_prompt`` prepended,
    so they're terse and the signal is clean. Pass ``verify_prompt=None`` to verify with
    your raw answers instead.

    Need the scores (consistency, per-answer agreement, the medoid)? Use
    :meth:`GroundyChecker.check` directly instead — that returns a :class:`GroundyResult`.
    """

    def decorate(func: Callable[..., str]) -> Callable[..., str]:
        checker = GroundyChecker(
            n=n,
            threshold=threshold,
            backend=backend,
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            reformulate_fn=reformulate_fn,
            verify_prompt=verify_prompt,
        )

        @functools.wraps(func)
        def wrapper(query: str, *args, **kwargs) -> str:
            if cache is not None:
                hit = cache.get(query)
                if hit is not None:
                    logger.debug("✅ cache HIT  | query={!r}", query)
                    return hit
                logger.debug("❌ cache MISS | query={!r} — running check", query)

            def answer_fn(q: str) -> str:
                return func(q, *args, **kwargs)

            result = checker.check(query, answer_fn)
            answer = result.best_answer if result.is_reliable else on_unreliable

            if cache is not None:
                cache.set(query, answer)
                logger.debug("💾 cache SET  | query={!r}", query)

            return answer

        return wrapper

    # bare @groundy vs called @groundy(...)
    if fn is not None and callable(fn):
        return decorate(fn)
    return decorate
