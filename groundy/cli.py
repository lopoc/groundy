"""
groundy.cli
~~~~~~~~~~~
A tiny ``groundy`` CLI — paste a question, watch it think, see the verdict.

Zero extra deps: hand-rolled ANSI + a braille spinner, both TTY-aware (they degrade to
plain text when piped). It reformulates *and* answers on ``GROUNDY_MODEL`` via the
OpenAI-compatible client, so ``GROUNDY_API_KEY`` + ``GROUNDY_MODEL`` runs the whole thing.

The CLI owns its output: it silences groundy's debug logging (even if ``GROUNDY_DEBUG=1``)
so the pretty render stays clean — pass ``--debug`` to see the raw reformulation/answer log.

    groundy "Who proved Fermat's Last Theorem?"
    echo "your question" | groundy
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time

# Quiet the embedding model's first-load chatter — huggingface_hub's "unauthenticated
# requests" warning and its "Loading weights" progress bar — so they don't stomp on the
# spinner. Must be set before sentence-transformers/huggingface_hub import; setdefault keeps
# any value the user already exported. (This noise is third-party, not groundy's own log.)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ANSI colours, switched off when piped or when NO_COLOR is set (https://no-color.org).
_PLAIN = bool(os.getenv("NO_COLOR")) or not sys.stdout.isatty()


def _paint(text: str, *codes: str) -> str:
    return text if _PLAIN else "".join(codes) + text + "\033[0m"


BOLD, DIM, GREEN, YELLOW, RED, CYAN, GREY = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[36m",
    "\033[90m",
)
# 24-bit violet (#A855F7) for the histogram bars — fancier than flat ANSI magenta.
VIOLET = "\033[38;2;168;85;247m"

# One refusal string, shared by the scatter view and -q (kept in step with the library's).
REFUSAL = "I'm not confident enough to answer that reliably."

# --matrix view: a-z row/column labels and a 5-step shade ramp (faint = low similarity,
# solid = high), so the pairwise agreement structure reads as bright blocks.
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
RAMP = "·░▒▓█"


class _Spinner:
    """A transient braille spinner. No-op when stdout isn't a TTY (e.g. piped)."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text: str):
        self.text = text
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if not _PLAIN:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {_paint(frame, CYAN)} {_paint(self.text, DIM)}")
            sys.stdout.flush()
            time.sleep(0.08)

    def __exit__(self, *_):
        if self._thread is None:  # plain/piped mode — nothing was ever drawn
            return
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.text) + 6) + "\r")  # wipe the line
        sys.stdout.flush()


def _bar(share: float, width: int = 8) -> str:
    """A tiny block meter: ███████░"""
    filled = round(max(0.0, min(1.0, share)) * width)
    return "█" * filled + "░" * (width - filled)


def _truncate(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _distinct(answers: list[str], agreement_scores: list[float]):
    """Pair each answer with its *agreement* — how well it agrees with the others, the very
    signal groundy scores on. Exactly-identical answers (the one equivalence we can assert
    without guessing) collapse into one row with a count. Returns ``[(agreement, count,
    answer)]``, strongest agreement first: the consensus on top, the outliers at the bottom.
    """
    rows: dict[str, list] = {}  # normalised text -> [agreement, count, original]
    for answer, fit in zip(answers, agreement_scores):
        key = " ".join(answer.split()).lower()
        if key in rows:
            rows[key][1] += 1
        else:
            rows[key] = [fit, 1, answer]
    return sorted(rows.values(), key=lambda r: r[0], reverse=True)


def _matrix(n: int, similarity_scores: list[float]) -> list[list[float]]:
    """Rebuild the full symmetric pairwise matrix (diagonal 1.0) from the flat upper triangle
    in the result — the raw substrate the consistency score is the average of."""
    m = [[1.0] * n for _ in range(n)]
    for (i, j), s in zip(itertools.combinations(range(n), 2), similarity_scores):
        m[i][j] = m[j][i] = s
    return m


def _cell(s: float) -> str:
    """A 2-char heat cell, shaded by similarity (clamped to [0, 1])."""
    return RAMP[round(max(0.0, min(1.0, s)) * (len(RAMP) - 1))] * 2


def _render(r, matrix: bool = False) -> None:
    """Pretty-print a GroundyResult: verdict, answer, then the agreement scatter (or matrix)."""
    ok = r.is_reliable
    mark = _paint("✓ reliable", GREEN, BOLD) if ok else _paint("⚠ uncertain", YELLOW, BOLD)
    score = _paint(f"consistency {r.consistency_score:.2f}", GREEN if ok else YELLOW)
    timing = _paint(f"{r.latency_ms / 1000:.1f}s", GREY)
    print(f"\n  {mark}   {score}   {_paint('·', GREY)} {timing}\n")

    if ok:
        print(f"  {_paint(r.best_answer, BOLD)}\n")
    else:
        print(f"  {_paint(REFUSAL, YELLOW)}\n")

    # The scatter — groundy's pairwise signal, shown two ways. Default: each distinct answer
    # with a bar = how much it agrees with the rest (consensus tall, outliers short). With
    # --matrix: the raw N×N heatmap, where mutually-agreeing answers light up as bright blocks
    # and the eye finds the clusters — no threshold, nothing aggregated.
    print(f"  {_paint('scatter', DIM)}")
    if matrix:
        labels = [ALPHABET[i] for i in range(len(r.answers))]
        m = _matrix(len(r.answers), r.similarity_scores)
        # Cells touch (no gap) so a cluster reads as one solid block, not vertical stripes.
        print("       " + "".join(f"{_paint(c, GREY)} " for c in labels))
        for i, answer in enumerate(r.answers):
            cells = "".join(_paint(_cell(m[i][j]), VIOLET) for j in range(len(r.answers)))
            print(f"    {_paint(labels[i], GREY)}  {cells}  {_truncate(answer, 46)}")
    else:
        for fit, count, answer in _distinct(r.answers, r.agreement_scores):
            bar = _paint(_bar(fit), VIOLET)
            tag = _paint(f" ×{count}", GREY) if count > 1 else ""
            print(f"    {bar} {_paint(f'{fit:.2f}', GREY)}  {_truncate(answer)}{tag}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="groundy",
        description="Ask a question several ways; pass only if the model agrees with itself.",
        epilog="Needs GROUNDY_API_KEY, GROUNDY_BASE_URL, GROUNDY_MODEL "
        "(reformulates and answers on that model).",
    )
    parser.add_argument("query", nargs="?", help="the question (or pipe it via stdin)")
    parser.add_argument("-n", type=int, default=5, help="answers compared (default: 5)")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.75, help="reliable cutoff (default: 0.75)"
    )
    parser.add_argument("--model", default=None, help="reformulation model (else GROUNDY_MODEL)")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the answer")
    parser.add_argument(
        "--matrix", action="store_true", help="show the full N×N pairwise agreement heatmap"
    )
    parser.add_argument("--debug", action="store_true", help="show the raw reformulation log")
    args = parser.parse_args(argv)

    # Convenience: load a local .env if python-dotenv happens to be installed (it's a dev
    # extra, never required). Silently skip it otherwise.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    query = args.query
    if query is None and not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    if not query:
        parser.error("no query given (pass it as an argument or pipe it via stdin)")

    model = args.model or os.getenv("GROUNDY_MODEL")
    if not model:
        print(_paint("✗ set GROUNDY_MODEL (or pass --model) first.", RED), file=sys.stderr)
        return 2
    base_url = os.getenv("GROUNDY_BASE_URL")
    if not base_url:
        print(
            _paint("✗ set GROUNDY_BASE_URL (your provider endpoint) first.", RED), file=sys.stderr
        )
        return 2

    from loguru import logger
    from openai import OpenAI

    from groundy import GroundyChecker

    # The CLI renders its own output, so groundy's debug log would just be noise — keep it
    # off even if GROUNDY_DEBUG=1 is set in the env, unless --debug explicitly asks for it.
    if args.debug:
        logger.enable("groundy")
    else:
        logger.disable("groundy")

    client = OpenAI(base_url=base_url, api_key=os.getenv("GROUNDY_API_KEY"))

    # check() calls answer_fn once per "way" (n verify calls), then once more for the served
    # answer. We tick the counter *after* each call returns and live-update the spinner, so a
    # given "i/n" only appears once that answer is actually back — it advances with the
    # answers, never ahead of them. After the last verify answer, check() scores them pairwise
    # (a fast local batched op — slow only on the first run, while the embedding model loads),
    # which is when we flip to "comparing answers…". (The spinner thread reads .text each frame.)
    spinner = _Spinner("reformulating…")
    answered = 0

    def answer_fn(q: str) -> str:
        nonlocal answered
        if answered >= args.n:  # all n verify answers are in — this is the served call
            spinner.text = "writing the answer…"
        msg = client.chat.completions.create(
            model=model, max_tokens=512, messages=[{"role": "user", "content": q}]
        )
        answered += 1
        if answered < args.n:
            spinner.text = f"asking {answered}/{args.n} ways…"
        elif answered == args.n:  # last verify in — check() now scores the answers pairwise
            spinner.text = "comparing answers…"
        return msg.choices[0].message.content

    checker = GroundyChecker(n=args.n, threshold=args.threshold, model=model, base_url=base_url)

    if not args.quiet:
        print(f"\n{_paint('🌱 groundy', GREEN, BOLD)}\n\n  {_paint('?', CYAN)} {query}")

    try:
        with spinner:
            result = checker.check(query, answer_fn)
    except Exception as e:  # noqa: BLE001 — surface any provider/parse error cleanly
        print(_paint(f"✗ {type(e).__name__}: {e}", RED), file=sys.stderr)
        return 1

    if args.quiet:
        print(result.best_answer if result.is_reliable else REFUSAL)
    else:
        _render(result, args.matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
