# Changelog

All notable changes to `groundy` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0: the API may still
change between releases).

## [Unreleased]

### Breaking
- Reformulation config moved to groundy's own namespace: reads `GROUNDY_API_KEY` /
  `GROUNDY_BASE_URL` (instead of `OPENAI_API_KEY` / `OPENAI_BASE_URL`), passed explicitly to
  the client — no OpenAI-branded env vars.
- `base_url` is now **required** (like `model`): pass `base_url=` or set `GROUNDY_BASE_URL`,
  else `__init__` raises. No silent default to OpenAI's endpoint — name your provider.
- Removed the `reformulate_fn` dependency-injection hook. Almost every provider is
  OpenAI-compatible, so `model`/`base_url`/`api_key` already covers them; the hook added
  config surface without shedding the `openai` dependency. Non-breaking to re-add later.

### Added
- `groundy` CLI — a zero-dep terminal vibe-check that asks a question a few ways and prints
  the verdict plus the *scatter*: each distinct answer with a bar = how much it agrees with
  the rest (`--matrix` for the full N×N pairwise heatmap). Reads `GROUNDY_API_KEY` +
  `GROUNDY_BASE_URL` + `GROUNDY_MODEL`; supports stdin, `-q`, `-n`, `-t`, `--matrix`,
  `--debug`. Silences groundy's debug log by default so the render stays clean.

## [0.1.0b1] — 2026-06-07

Configuration simplified to a single provider: groundy now needs just an API key, a
provider, and a model name, read like any OpenAI client.

### Breaking
- Renamed env `GROUNDY_REFORMULATION_MODEL` → `GROUNDY_MODEL`; the reformulation model is
  now required (no `gpt-4o-mini` fallback) — set `model=` or `GROUNDY_MODEL` or it raises.

### Fixed
- `best_answer` returns the served answer (not the consensus/medoid) when reliable, else
  `None`; docs now match the code.

## [0.1.0b0]

- Initial release: the `@groundy` decorator over an OpenAI-compatible SDK.

### Added
- Initial release: the `@groundy` decorator, `GroundyChecker.check()` rich result, the
  `Cache` protocol with `cache=` orchestration, pluggable similarity backends
  (`embeddings` real, `llm_judge` stub), and injectable `reformulate_fn`.
