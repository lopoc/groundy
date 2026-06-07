"""
groundy.core
~~~~~~~~~~~~
Hallucination detection via semantic self-consistency.

Ask the same question several ways. If the answers agree, the model is confident; if they
scatter, it's improvising — so return a refusal instead. No ground truth, no fine-tuning,
no retrieval.

The headline API is the ``@groundy`` decorator: it turns a ``query -> str`` LLM call into
a *trustworthy* ``query -> str`` — same signature, but the answer is either one the model
agrees with itself on, or a refusal string.
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from loguru import logger
from openai import OpenAI

from groundy.prompts import REFORMULATION_SYSTEM, REFORMULATION_USER

# groundy makes ONE LLM call of its own — reformulation. It needs just three things, read
# like any OpenAI client: an API key (OPENAI_API_KEY), a provider (OPENAI_BASE_URL), and a
# model name (model= or GROUNDY_MODEL). There is no default model — you name it explicitly.

# Reformulations use temperature 0 so the rephrasings are stable across runs.
TEMPERATURE = 0.0

# Returned (and cached) in place of an answer when the model disagrees with itself.
DEFAULT_REFUSAL = "I'm not confident enough to answer that reliably."

# Prepended to each query when producing the *verify* answers — the ones compared for
# consistency — so divergence is about substance, not phrasing or length. The answer
# actually served is produced separately, exactly as you wrote your function.
VERIFY_PROMPT = (
    "Answer as briefly as possible — just the answer itself, "
    "with no explanation, no caveats, and no preamble."
)


@runtime_checkable
class Cache(Protocol):
    """Minimal cache interface ``@groundy(cache=...)`` drives.

    Any object with these two methods works — a dict, Redis, a managed semantic cache.
    groundy never stores, embeds, or evicts; retention/TTL/eviction stay the cache's job.
    """

    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str) -> None: ...


@dataclass
class GroundyResult:
    """Rich result from :meth:`GroundyChecker.check`."""

    original_query: str
    original_answer: str  # the SERVED answer; "" when unreliable (only produced if reliable)
    reformulations: list[str]
    answers: list[str]  # the terse VERIFY answers, aligned to [query, *reformulations]
    similarity_scores: list[float]  # pairwise, over the verify answers
    consistency_score: float  # mean pairwise similarity, 0-1
    is_reliable: bool  # consistency_score >= threshold
    threshold: float
    backend: str
    latency_ms: float
    # Consensus = the verify answer agreeing most with the rest (the medoid). Diagnostic
    # only — best_answer returns the served answer, not this.
    consensus_answer: str = ""
    consensus_index: int = 0
    agreement_scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def best_answer(self) -> Optional[str]:
        """The served answer when reliable, else None (the decorator maps None → refusal)."""
        return self.original_answer if self.is_reliable else None

    def __repr__(self):
        status = "✓ RELIABLE" if self.is_reliable else "⚠ UNCERTAIN"
        return (
            f"GroundyResult({status} | consistency={self.consistency_score:.3f} | "
            f"threshold={self.threshold} | n={len(self.reformulations) + 1})"
        )


class GroundyChecker:
    """Core checker. Use directly for the rich :class:`GroundyResult`, or use the
    ``@groundy`` decorator for the simple ``str``-in/``str``-out path.

    groundy's own (reformulation) call needs just an API key and a provider, read like any
    OpenAI client — ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` — so it works against any
    OpenAI-compatible provider. Your *answer* call is whatever ``answer_fn`` does; the two
    are independent and can use different models or providers.

    Parameters
    ----------
    n : int
        Answers compared: the original query's answer plus n-1 reformulation answers. Must
        be >= 2. Higher = sturdier and pricier; 3-5 is a good range.
    threshold : float
        Minimum consistency score (0-1) to call the answer reliable. Calibrate it.
    backend : str
        Similarity backend: ``'embeddings'`` (local, default) or ``'llm_judge'`` (stub).
    model, temperature, base_url, api_key :
        Provider config for the default reformulator. ``model`` is required (no default):
        ``model=`` → ``GROUNDY_MODEL``, else a ``ValueError``. ``base_url``/``api_key`` are
        left None when unset so the OpenAI SDK reads ``OPENAI_*``. All ignored if
        ``reformulate_fn`` is set.
    reformulate_fn : Callable[[str, int], list[str]] | None
        Bring your own reformulator — ``(query, k) -> k rephrasings`` — and the library
        touches no vendor SDK. When None, the OpenAI-compatible default is used.
    verify_prompt : str | None
        Prepended to each query for the verify answers (forces terseness). None verifies
        with your raw answers instead.
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
            raise ValueError(f"n must be >= 2 (need 2+ answers to compare), got {n}.")
        self.n = n
        self.threshold = threshold
        self.backend = backend
        # groundy's own reformulation call: just a model + provider. base_url/api_key stay
        # None when unset so the OpenAI SDK reads OPENAI_BASE_URL / OPENAI_API_KEY itself.
        self.model = model or os.getenv("GROUNDY_MODEL")
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.reformulate_fn = reformulate_fn
        self.verify_prompt = verify_prompt

        # The default reformulator needs a model name — there's no fallback. (Not required
        # when you inject reformulate_fn: then groundy makes no LLM call of its own.)
        if reformulate_fn is None and not self.model:
            raise ValueError(
                "No reformulation model. Pass model= or set GROUNDY_MODEL "
                "(or inject reformulate_fn to skip groundy's own LLM call)."
            )

        self._similarity_fn = self._load_backend(backend)
        self._client: Optional[OpenAI] = None  # built on first default-reformulator use

    def _load_backend(self, backend: str) -> Callable:
        if backend == "embeddings":
            from groundy.backends.embeddings import cosine_similarity_batch

            return cosine_similarity_batch
        if backend == "llm_judge":
            from groundy.backends.llm_judge import judge_similarity_batch

            return judge_similarity_batch
        raise ValueError(f"Unknown backend: {backend!r}. Use 'embeddings' or 'llm_judge'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, query: str, answer_fn: Callable[[str], str]) -> GroundyResult:
        """Run the full consistency check and return a rich :class:`GroundyResult`.

        ``answer_fn`` is your LLM call (``query -> str``). **It must hit the raw model**: if
        a *semantic* cache sits underneath, the reformulations collapse to one cached
        answer, similarity is trivially 1.0, and every check falsely passes.

        It is called in two modes: once per ``[query, *reformulations]`` with
        ``verify_prompt`` prepended (the terse *verify* answers, compared for consistency),
        and — only if the verdict is reliable — once on the raw query (the *served*
        answer). The served call is skipped on an unreliable check to save tokens, so
        ``original_answer`` is ``""`` then.
        """
        t0 = time.monotonic()
        logger.debug(
            "🔍 check start | n={} threshold={} backend={}",
            self.n,
            self.threshold,
            self.backend,
        )
        logger.debug("💬 query | {!r}", query)

        # 1. Reformulate the query n-1 times (injected reformulator, else the default one).
        k = self.n - 1
        if self.reformulate_fn is not None:
            reformulations = list(self.reformulate_fn(query, k))
        else:
            reformulations = self._generate_reformulations(query)

        # 2. Verify answers — terse, so consistency is about substance, not phrasing.
        verify_inputs = [query] + reformulations
        answers = []
        for i, q in enumerate(verify_inputs, start=1):
            vq = f"{self.verify_prompt}\n\n{q}" if self.verify_prompt else q
            a = answer_fn(vq)
            answers.append(a)
            logger.debug("📝 verify {}/{} | {!r} -> {!r}", i, len(verify_inputs), q, a)

        # 3. Pairwise similarity across the verify answers.
        index_pairs = list(itertools.combinations(range(len(answers)), 2))
        similarity_scores = self._similarity_fn(
            [answers[i] for i, _ in index_pairs],
            [answers[j] for _, j in index_pairs],
        )

        # 4. Consistency = mean pairwise similarity; reliable if it clears the threshold.
        consistency_score = (
            sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        )
        is_reliable = consistency_score >= self.threshold

        # 5. Consensus (medoid) verify answer — the one agreeing most. Diagnostic only.
        agreement_scores = self._agreement_scores(len(answers), index_pairs, similarity_scores)
        consensus_index = (
            max(range(len(answers)), key=agreement_scores.__getitem__) if answers else 0
        )
        consensus_answer = answers[consensus_index] if answers else ""

        verdict = "✅ reliable" if is_reliable else "⚠️ uncertain"
        logger.debug("📊 result | consistency={:.3f} {}", consistency_score, verdict)

        # 6. Served answer — produced last and only if reliable (else we'd discard it).
        if is_reliable:
            original_answer = answer_fn(query)
            logger.debug("📤 served | {!r}", original_answer)
        else:
            original_answer = ""
            logger.debug("⏭️ served skipped — uncertain")

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
            latency_ms=(time.monotonic() - t0) * 1000,
            consensus_answer=consensus_answer,
            consensus_index=consensus_index,
            agreement_scores=agreement_scores,
        )

    @staticmethod
    def _agreement_scores(n, index_pairs, similarity_scores):
        """Per-answer mean similarity to the other n-1 answers. The highest is the medoid."""
        if n <= 1:
            return [0.0] * n
        sums = [0.0] * n
        for (i, j), s in zip(index_pairs, similarity_scores):
            sums[i] += s
            sums[j] += s
        return [sums[k] / (n - 1) for k in range(n)]

    def _generate_reformulations(self, query: str) -> list[str]:
        """Ask the model for n-1 semantically equivalent rephrasings (a JSON array)."""
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        kwargs = dict(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": REFORMULATION_SYSTEM},
                {
                    "role": "user",
                    "content": REFORMULATION_USER.format(n=self.n - 1, query=query),
                },
            ],
        )
        # Pin temperature for reproducibility; omit it when None (some models reject it).
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        raw = self._client.chat.completions.create(**kwargs).choices[0].message.content.strip()
        # Strip a ```json fence if the model added one.
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
    """Turn a ``query -> str`` LLM call into a *trustworthy* ``query -> str``.

    The wrapped function keeps its signature and still returns a ``str`` — either an answer
    the model agrees with itself on, or ``on_unreliable`` when it doesn't. Works bare or
    called::

        @groundy
        def ask(q: str) -> str: ...

        @groundy(threshold=0.8, cache=my_cache)
        def ask(q: str) -> str: ...

    The wrapped call must hit the **raw** model — see :meth:`GroundyChecker.check`. Pass
    ``cache=`` (any object with ``get``/``set``) to run only on a miss. groundy's own
    reformulation call needs just an API key and a provider (``OPENAI_*``); configure it
    with ``model``/``base_url``/``api_key`` or swap it out entirely with ``reformulate_fn``.
    Want the scores? Use :meth:`GroundyChecker.check` directly.
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
                    logger.debug("✅ cache HIT  | {!r}", query)
                    return hit
                logger.debug("❌ cache MISS | {!r} — running check", query)

            result = checker.check(query, lambda q: func(q, *args, **kwargs))
            answer = result.best_answer if result.is_reliable else on_unreliable

            if cache is not None:
                cache.set(query, answer)
                logger.debug("💾 cache SET  | {!r}", query)
            return answer

        return wrapper

    # bare @groundy vs called @groundy(...)
    return decorate(fn) if callable(fn) else decorate
