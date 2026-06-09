# Changelog

All notable changes to `groundy` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0: the API may still
change between releases).

## [0.2.0] — 2026-06-08

### Breaking
- Env config moved to groundy's own namespace: `GROUNDY_API_KEY` / `GROUNDY_BASE_URL`
  (was `OPENAI_*`).
- `base_url` is now required (like `model`) — no default provider.
- Removed the `reformulate_fn` hook; any OpenAI-compatible `base_url` covers it.

### Added
- `groundy` CLI — terminal vibe-check printing the verdict + agreement matrix
  (`--matrix` for the N×N heatmap). Supports stdin and `-q`/`-n`/`-t`/`--debug`.

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
