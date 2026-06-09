"""
groundy.core
~~~~~~~~~~~~
Hallucination detection via semantic self-consistency.

Ask the same question several ways: if the answers agree the model is confident, if they
scatter it's improvising — so return a refusal instead. No ground truth, no fine-tuning, no
retrieval. The headline API is the ``@groundy`` decorator; ``GroundyChecker.check()`` is the
rich-result door underneath it.
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from loguru import logger
from openai import OpenAI

from groundy.prompts import REFORMULATION_SYSTEM, REFORMULATION_USER

# groundy makes ONE LLM call of its own — reformulation, over an OpenAI-compatible API. It
# needs just three things, all under its own namespace: an API key (GROUNDY_API_KEY), a
# provider (GROUNDY_BASE_URL), and a model name (model= or GROUNDY_MODEL). Model and provider
# have no default — you name them explicitly. Point them at any OpenAI-compatible endpoint.

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
    """Core checker — ``check()`` returns a rich :class:`GroundyResult`. (For the simple
    ``str``-in/``str``-out path, use the ``@groundy`` decorator.) See the README for the
    full story; params below are the reference.

    n : answers compared (original + n-1 reformulations), >= 2.
    threshold : min consistency score (0-1) to call an answer reliable.
    backend : similarity backend — ``'embeddings'`` (sentence-transformers, local, default),
        ``'fastembed'`` (same model via ONNX, ~15x lighter import, needs the fastembed extra),
        or ``'llm_judge'`` (stub).
    model, temperature, base_url, api_key : config for the reformulation call. ``model`` and
        ``base_url`` are **required** (kwarg → ``GROUNDY_MODEL`` / ``GROUNDY_BASE_URL``, else
        ``ValueError`` — no default provider); ``api_key`` → ``GROUNDY_API_KEY`` (may be unset
        for keyless local servers). Any OpenAI-compatible endpoint works.
    verify_prompt : prepended to the verify answers to force terseness (None to skip).
    concurrency : how many of the n verify ``answer_fn`` calls run at once (default 2; 1 =
        sequential). They're independent, so this cuts wall-clock; the served call stays
        sequential. ``answer_fn`` must be thread-safe when > 1 (a plain LLM/HTTP call is).
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
        verify_prompt: Optional[str] = VERIFY_PROMPT,
        concurrency: int = 2,
    ):
        if n < 2:
            raise ValueError(f"n must be >= 2 (need 2+ answers to compare), got {n}.")
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}.")
        self.n = n
        self.concurrency = concurrency
        self.threshold = threshold
        self.backend = backend
        # groundy's own reformulation call: a model + provider, all under the GROUNDY_*
        # namespace (no OpenAI-branded env vars). kwargs win over env; either may be unset.
        self.model = model or os.getenv("GROUNDY_MODEL")
        self.temperature = temperature
        self.base_url = base_url or os.getenv("GROUNDY_BASE_URL")
        self.api_key = api_key or os.getenv("GROUNDY_API_KEY")
        self.verify_prompt = verify_prompt

        # The reformulation call needs a model and a provider — no defaults, name them.
        if not self.model:
            raise ValueError("No reformulation model. Pass model= or set GROUNDY_MODEL.")
        if not self.base_url:
            raise ValueError("No reformulation provider. Pass base_url= or set GROUNDY_BASE_URL.")

        self._similarity_fn = self._load_backend(backend)
        self._client: Optional[OpenAI] = None  # built on first reformulation call

    def _load_backend(self, backend: str) -> Callable:
        if backend == "embeddings":
            from groundy.backends.embeddings import cosine_similarity_batch

            return cosine_similarity_batch
        if backend == "fastembed":
            from groundy.backends.fastembed import cosine_similarity_batch

            return cosine_similarity_batch
        if backend == "llm_judge":
            from groundy.backends.llm_judge import judge_similarity_batch

            return judge_similarity_batch
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'embeddings', 'fastembed', or 'llm_judge'."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, query: str, answer_fn: Callable[[str], str]) -> GroundyResult:
        """Run the full consistency check and return a rich :class:`GroundyResult`.

        ``answer_fn`` is your ``query -> str`` LLM call. **It must hit the raw model** — a
        semantic cache underneath collapses the reformulations to one answer and every check
        falsely passes (see README). It's called for the terse *verify* answers, then once
        more on the raw query for the *served* answer — only if reliable, else
        ``original_answer`` stays ``""``.
        """
        t0 = time.monotonic()
        logger.debug(
            "🔍 check start | n={} threshold={} backend={}",
            self.n,
            self.threshold,
            self.backend,
        )
        logger.debug("💬 query | {!r}", query)

        # 1. Reformulate the query n-1 times.
        reformulations = self._generate_reformulations(query)

        # 2. Verify answers — terse, so consistency is about substance, not phrasing. The n
        # calls are independent, so run up to self.concurrency at once (order preserved, so
        # answers[i] stays aligned to verify_inputs[i] for the pairwise indexing below).
        verify_inputs = [query] + reformulations

        def _verify(q: str) -> str:
            vq = f"{self.verify_prompt}\n\n{q}" if self.verify_prompt else q
            return answer_fn(vq)

        if self.concurrency > 1 and len(verify_inputs) > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                answers = list(pool.map(_verify, verify_inputs))
        else:
            answers = [_verify(q) for q in verify_inputs]

        for i, (q, a) in enumerate(zip(verify_inputs, answers), start=1):
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
    verify_prompt: Optional[str] = VERIFY_PROMPT,
    concurrency: int = 2,
    cache: Optional[Cache] = None,
    on_unreliable: str = DEFAULT_REFUSAL,
):
    """Turn a ``query -> str`` LLM call into a *trustworthy* ``query -> str`` — same
    signature, but the answer is one the model agrees with itself on, or ``on_unreliable``.
    Works bare (``@groundy``) or called (``@groundy(threshold=0.8, cache=my_cache)``).

    The wrapped call must hit the **raw** model. Pass ``cache=`` to run only on a miss.
    Want the scores instead? Use :meth:`GroundyChecker.check`. See the README for the rest.
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
            verify_prompt=verify_prompt,
            concurrency=concurrency,
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
