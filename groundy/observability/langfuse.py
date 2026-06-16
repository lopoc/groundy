"""
groundy.observability.langfuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Langfuse adapter for groundy's :class:`~groundy.observability.Tracer` protocol. Install the
extra (``pip install groundy[langfuse]`` / ``uv add groundy[langfuse]``); the core never
imports this module — only you do, when you wire it up::

    from groundy.observability.langfuse import LangfuseTracer

    @groundy(tracer=LangfuseTracer())
    def ask(q: str) -> str:
        ...

Requires ``langfuse>=3.2`` (the unified ``start_observation`` API). The Langfuse client
reads its own credentials from the environment — ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_BASE_URL`` — unless you pass a pre-built ``client``.
"""

from __future__ import annotations


class _LangfuseSpan:
    """Wraps one Langfuse observation as a groundy :class:`~groundy.observability.Span`."""

    def __init__(self, obs, client, *, flush_on_end: bool = False):
        self._obs = obs
        self._client = client
        self._flush_on_end = flush_on_end
        self._ended = False

    def span(
        self, name, *, kind="span", input=None, metadata=None, model=None, model_parameters=None
    ):
        is_gen = kind == "generation"
        extra = {}
        if is_gen:  # model / model_parameters are generation-only in Langfuse
            if model is not None:
                extra["model"] = model
            if model_parameters:
                extra["model_parameters"] = model_parameters
        child = self._obs.start_observation(
            name=name,
            as_type="generation" if is_gen else "span",
            input=input,
            metadata=metadata or None,
            **extra,
        )
        return _LangfuseSpan(child, self._client)

    def end(self, *, output=None, usage=None, metadata=None):
        if self._ended:
            return
        # Only forward what we actually have — usage_details is generation-only, and an
        # empty dict would needlessly overwrite.
        update = {}
        if output is not None:
            update["output"] = output
        if usage:
            update["usage_details"] = usage
        if metadata:
            update["metadata"] = metadata
        if update:
            self._obs.update(**update)
        self._obs.end()
        self._ended = True
        if self._flush_on_end:
            self._client.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and not self._ended:
            self._obs.update(level="ERROR", status_message=str(exc))
        self.end()
        return False


class LangfuseTracer:
    """A groundy ``Tracer`` backed by Langfuse.

    Pass a pre-built ``client`` to reuse one (and its config); otherwise a ``Langfuse()`` is
    constructed here, reading credentials from the environment. ``flush_on_end``
    flushes the client when each *root* trace closes — safe for short-lived scripts that would
    otherwise exit before Langfuse's background flush; set ``False`` to rely on Langfuse's own
    batching when you're tracing many checks in a long-running process.
    """

    def __init__(self, client=None, *, flush_on_end: bool = True):
        if client is None:
            from langfuse import Langfuse

            client = Langfuse()
        self._client = client
        self._flush_on_end = flush_on_end

    def trace(self, name, *, input=None, metadata=None):
        root = self._client.start_observation(
            name=name, as_type="span", input=input, metadata=metadata or None
        )
        return _LangfuseSpan(root, self._client, flush_on_end=self._flush_on_end)
