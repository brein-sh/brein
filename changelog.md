# Changelog

All notable changes to brein are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

A push to `main` that adds a new `## [X.Y.Z] - YYYY-MM-DD` heading is auto-tagged `vX.Y.Zf` and published by `publish.yml`. Tags ending in `f` skip tests (force release).

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
