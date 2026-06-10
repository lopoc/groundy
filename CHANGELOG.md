# Changelog

Notable changes to `groundy`. Pre-1.0, so the API may still shift between releases.

## 0.3.0 — 2026-06-11

- Added agnostic observability: pass a `tracer` to `@groundy` / `GroundyChecker` (a small
  `Tracer` protocol, like `cache=`) and each `check()` emits a nested trace —
  `reformulate → verify ×n → score → served`. Default `tracer=None` is a no-op (zero overhead).
- Ships a Langfuse adapter behind the `groundy[langfuse]` extra
  (`from groundy.observability.langfuse import LangfuseTracer`); the core imports no vendor SDK.
  The reformulation node (the one call groundy owns) carries the model, temperature, token
  usage, and a prompt-template hash.

## 0.2.1 — 2026-06-09

- Added a `fastembed` similarity backend: the same `all-MiniLM-L6-v2` model via ONNX
  Runtime (no torch), ~15x lighter import (CLI cold start ~10s → ~1-2s). Opt-in via
  `backend="fastembed"` + the `fastembed` extra; the CLI defaults to it and falls back to
  `embeddings` when it isn't installed.
- Added a `concurrency` knob (`GroundyChecker` / `@groundy`, default 2; CLI `-c`) to fetch
  the verify answers in parallel — cuts wall-clock since they're independent. The served
  call stays sequential.

## 0.2.0 — 2026-06-08

- **Breaking:** env config moved to groundy's own namespace — `GROUNDY_API_KEY` /
  `GROUNDY_BASE_URL` (was `OPENAI_*`), and `base_url` is now required (no default provider).
- **Breaking:** removed the `reformulate_fn` hook — any OpenAI-compatible `base_url` covers it.
- Added the `groundy` CLI: a terminal vibe-check printing the verdict + agreement matrix
  (`--matrix` for the N×N heatmap). Supports stdin and `-q`/`-n`/`-t`/`--debug`.

## 0.1.0b1 — 2026-06-07

- **Breaking:** renamed env `GROUNDY_REFORMULATION_MODEL` → `GROUNDY_MODEL`; the
  reformulation model is now required (no `gpt-4o-mini` fallback) — set `model=` or
  `GROUNDY_MODEL` or it raises.
- Fixed: `best_answer` returns the served answer (not the consensus/medoid) when reliable,
  else `None`; docs now match the code.

## 0.1.0b0

- Initial release: the `@groundy` decorator, `GroundyChecker.check()` rich result, the
  `Cache` protocol with `cache=` orchestration, pluggable similarity backends (`embeddings`
  real, `llm_judge` stub), and an injectable `reformulate_fn`.
