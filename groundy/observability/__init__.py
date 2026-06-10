"""
groundy.observability
~~~~~~~~~~~~~~~~~~~~~~~
Agnostic tracing seam. ``GroundyChecker.check()`` emits a nested trace of every check —
reformulation, each verify answer, scoring, and the served answer — to a user-supplied
``Tracer``. This mirrors the :class:`~groundy.core.Cache` protocol: groundy drives the
interface but ships **no vendor SDK in the core**. A Langfuse adapter lives in
:mod:`groundy.observability.langfuse` behind the ``groundy[langfuse]`` extra — the core
never imports it.

The one inherent limitation: groundy owns only the *reformulation* LLM call, so that is the
sole node carrying token ``usage``. The verify/served nodes go through the user's
``answer_fn`` (``str`` in, ``str`` out) — groundy sees their text and timing, never their
model or tokens.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Span(Protocol):
    """One node in the trace tree.

    Open children with :meth:`span`, close with :meth:`end` (or a ``with`` block).
    ``end()`` is idempotent; ``__exit__`` ends the span and records the exception if the
    block raised. groundy runs the verify answers in a thread pool, so :meth:`span` may be
    called concurrently from multiple threads off the same parent — child creation must be
    thread-safe.

    ``kind`` is ``"span"`` for plain steps or ``"generation"`` for LLM calls; adapters may
    interpret other strings. ``model`` (the model name) and ``model_parameters`` (e.g.
    ``{"temperature": 0.0}``) apply only to generations — they let a tracer attribute the
    call to a model and surface its sampling params; ignored for plain spans. ``usage`` (on
    :meth:`end`) is a token-count dict, e.g. ``{"input": int, "output": int, "total": int}``,
    likewise generation-only.
    """

    def span(
        self,
        name: str,
        *,
        kind: str = "span",
        input: object = None,
        metadata: Optional[dict] = None,
        model: Optional[str] = None,
        model_parameters: Optional[dict] = None,
    ) -> "Span": ...

    def end(
        self,
        *,
        output: object = None,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None: ...

    def __enter__(self) -> "Span": ...

    def __exit__(self, exc_type, exc, tb) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """Opens the root span of a trace. ``check()`` defaults to :class:`NoopTracer`."""

    def trace(
        self, name: str, *, input: object = None, metadata: Optional[dict] = None
    ) -> Span: ...


class _NoopSpan:
    """A span that does nothing — the default, so ``check()`` can call the tracer
    unconditionally with zero overhead when none is configured."""

    def span(
        self, name, *, kind="span", input=None, metadata=None, model=None, model_parameters=None
    ):
        return self

    def end(self, *, output=None, usage=None, metadata=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class NoopTracer:
    """The default tracer: every call is a no-op."""

    def trace(self, name, *, input=None, metadata=None):
        return _NoopSpan()


__all__ = ["Span", "Tracer", "NoopTracer"]
