# Changelog

All notable changes to brein are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

A push to `main` that adds a new `## [X.Y.Z] - YYYY-MM-DD` heading is auto-tagged `vX.Y.Zf` and published by `publish.yml`. Tags ending in `f` skip tests (force release).

## [0.5.7] - 2026-06-27

### Security
- Secret scanner now blocks AWS access key IDs (`AKIA…` / `ASIA…`). Found via E2E test that submitted `AKIAIOSFODNN7EXAMPLE` and saw it committed.

### Added
- E2E coverage expanded to 10 tests over the real fastembed backend: semantic recall (lexically distant query), write→reindex→search closure, validator rollback assertion (file gone + HEAD unchanged), path-traversal block, secret-scanning block, CLI smoke (`brein doctor`, `brein index status`). Tests now drive the working tree (`python -m brain_mcp.server` / `brain_mcp.cli`) instead of the globally-installed binary.

## [0.5.6] - 2026-06-27

### Added
- E2E test suite (`tests/test_e2e.py`) driving the real MCP server over stdio: search round-trip, write loop (file → commit → push to bare remote), and **telemetry conservation** — every `brain_search` must emit exactly one `tool_call` row and one `search` row, with the documented schema. Catches silent telemetry regressions.
- `test.yml` workflow runs the suite on every push and PR.

### Changed
- Publish pipeline no longer falls through on missing tests; `pytest -q` is now blocking.

## [0.5.5] - 2026-06-27

### Fixed
- CI: write `.npmrc` manually instead of relying on `setup-node`'s auto-config. The auto-config emits `always-auth=true`, which causes granular npm tokens to be rejected with a misleading 404 on publish.

## [0.5.4] - 2026-06-27

### Changed
- CI: drop `--provenance` from `npm publish` (npm masks unrelated errors as 404 and our fallback couldn't distinguish them). Re-add once the publish pipeline is stable.

## [0.5.3] - 2026-06-27

### Changed
- PyPI package renamed from `brain-mcp` to `brein-mcp` (the original name was held by an unrelated maintainer). The `brain-mcp` binary and `brain_mcp` import path are unchanged.

## [0.5.2] - 2026-06-27

### Added
- Shared HTTP daemon. `brein daemon` runs one `brain-mcp` server over `streamable-http`; all MCP clients connect to the same URL so the embedder model is loaded once instead of once per client (~800MB × N → ~800MB total).
- `brein daemon launchd` prints a ready-to-load `~/Library/LaunchAgents/sh.brein.daemon.plist`.
- `brein daemon url` prints the daemon URL.
- `brein mcp <client> --http-url <url>` emits a client config that connects to the shared daemon instead of spawning stdio per client.

### Changed
- Server respects `BRAIN_MCP_TRANSPORT=http` (with `BRAIN_MCP_HOST` / `BRAIN_MCP_PORT`, default `127.0.0.1:8765`). Stdio remains the default for back-compat.

## [0.5.1] - 2026-06-27

### Added
- GitHub Actions release pipeline: pushing a new `## [X.Y.Z]` heading to `changelog.md` on `main` auto-tags `vX.Y.Zf`, which publishes `brain-mcp` to PyPI and `brein` to npm, then cuts a GitHub Release.

## [0.5.0] - 2026-06-27

### Added
- `brain_index_status` MCP tool for inspecting / restarting the embeddings index builder.
- Background consistency checker that runs on every brain write.

### Changed
- `brain_search` is now embeddings-only and returns a status payload (`building` / `stalled` / `missing` / `empty`) when the index is not ready, instead of falling back to grep.
- Hooks: orientation gate keys on a Read of the brain repo path (honors `$BRAIN_REPO`), not on `brain_search`.

### Fixed
- Validator: stale review-cycle is a warning, not a blocking error.
