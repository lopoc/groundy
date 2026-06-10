import os
from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from groundy.core import Cache, GroundyChecker, GroundyResult, groundy
from groundy.observability import NoopTracer, Span, Tracer

# Silent in production by default. Turn debug logging on for dev environments by
# setting GROUNDY_DEBUG=1 (e.g. in your dev .env) — never set it in production.
_DEBUG = os.getenv("GROUNDY_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
if _DEBUG:
    logger.enable("groundy")
else:
    logger.disable("groundy")

__all__ = [
    "groundy",
    "GroundyChecker",
    "GroundyResult",
    "Cache",
    "Tracer",
    "Span",
    "NoopTracer",
]

# Single source of truth: the version declared in pyproject.toml (read from the
# installed package metadata), so this never drifts from the distribution.
try:
    __version__ = version("groundy")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0+unknown"
