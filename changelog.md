# Changelog

All notable changes to brein are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

A push to `main` that adds a new `## [X.Y.Z] - YYYY-MM-DD` heading is auto-tagged `vX.Y.Zf` and published by `publish.yml`. Tags ending in `f` skip tests (force release).

## [0.5.17] - 2026-06-28

### Fixed
- **Consistency worker spawned the wrong binary when launched from the daemon.** `_brein_executable()` preferred `sys.argv[0]` (path-exists check) over `shutil.which("brein")`. When the daemon forks the consistency worker, `sys.argv[0]` is `/usr/local/bin/brain-mcp` (the MCP **server** launcher), not `brein` (the CLI). The launcher ignored `consistency check <path>` as args and tried to start a second MCP HTTP server, failed to bind to the already-occupied 8765, and exited as `<defunct>`. No consistency check ever ran in daemon mode. Now we prefer `which("brein")` first and only fall back to `sys.argv[0]` if its basename actually is `brein`. Caught by manually triggering a brain_update against the v0.5.16 daemon and finding the spawned PID was a zombie that had logged "address already in use" instead of an agent run.

## [0.5.16] - 2026-06-28

### Changed
- **Consistency checker is now agentic and actually resolves contradictions.** Previously the post-write consistency worker was a one-shot LLM judge that wrote `{kind, suggested_fix, related_paths}` into a queue file and stopped. Nothing acted on `auto_merge` or `contradiction` findings; nothing ever picked a canonical winner; the queue just grew until someone called `brain_consistency_status`. Now the worker invokes the LLM with `Read,Grep,Glob,Edit` tool access (claude `--allowed-tools`), gives it the brain repo as cwd, and asks it to either RESOLVE or ESCALATE. When the agent picks a clear winner (`source_of_truth: true` OR newer `last_reviewed`/`decided` date), it edits the loser docs to add a `> **Superseded by [[canonical]].**` line, then the worker commits + pushes under the same `_interprocess_write_lock` brain_update uses. When the agent isn't sure, it escalates — `kind: "escalate"` lands in the queue with an `escalation_reason`. Auto-resolved findings still emit so you can audit.

### Refactored
- **Single source of truth for LLM invocation.** Previously `eval.py` had a multi-CLI selector (claude → codex → gemini → OpenRouter fallback) while `consistency.py` hardcoded `claude` and required users to set `BRAIN_JUDGE_CMD` to use anything else — Codex- and Cursor-only users got no consistency checking at all. Factored both onto `shared.ask_llm(prompt, *, disable_brain, allowed_tools, cwd, timeout_s)`. Consistency now inherits the same fallback chain. `allowed_tools` is the agentic switch — when set on claude, passes `--allowed-tools`; on other CLIs it's ignored (one-shot).

## [0.5.15] - 2026-06-28

### Fixed
- **Concurrent-fire dedup race in the eval worker.** The 24h `_seen_recently` check was read-then-decide with the LLM gate (~3-5s) sitting between the read and the `_mark_seen` write. Two workers spawned within that window both saw "not seen" → both passed the gate → both ran the full A/B → wasted ~$2 + 2.5 min per duplicate. Now we use an O_EXCL claim slot (`~/.brein/eval-claims/<hash>.claim`) before the gate runs. The second worker bails atomically. Stale claims (>10 min, configurable via `BRAIN_EVAL_CLAIM_STALE_SECONDS`) are swept so a crashed worker doesn't block forever.

## [0.5.14] - 2026-06-27

### Fixed
- `UserPromptSubmit` hook now extracts the actual user prompt from Claude Code's JSON envelope (which also carries `session_id`, `transcript_path`, etc.). Before this fix the eval worker hashed the entire envelope — so every prompt looked novel (defeating dedup) and the question A/B'd was the literal JSON, not the question. New CLI: `brein eval capture-prompt --out PATH`.

### Added
- `BRAIN_OBSERVE_PATHS` env var (colon-separated) lets `eval observe` benchmark Grep/Read into additional brain repo clones beyond the primary `$BRAIN_REPO`. Useful when you have e.g. a legacy `~/.braincorp/brain` alongside the canonical `~/Documents/GitHub/company-brain`.

## [0.5.13] - 2026-06-27

### Fixed
- `brein mcp <client> --http-url` now emits `{type: "http", url: ...}` instead of just `{url: ...}`. Newer Claude Code releases require the explicit `type` field — without it the server was parsed into `~/.claude.json` but never appeared in the `/mcp` UI, even though the daemon was healthy on `127.0.0.1:8765`.

## [0.5.12] - 2026-06-27

### Added
- **Eval now benchmarks plain `Grep`/`Read`/`Glob` against the brain repo**, not just MCP tool calls. Two new Claude Code hooks:
  - `UserPromptSubmit` captures the prompt to `/tmp/claude-brein-last-prompt-$SESSION`.
  - `PostToolUse` on `Read|Grep|Glob` runs `brein eval observe`, which checks whether the tool targeted a path under `$BRAIN_REPO` and — if so — spawns a detached eval worker using the saved prompt as the question. Same LLM gate, dedup, and conditional A/B as the MCP-tool path.
  - Closes the gap from 0.5.11: even when Claude routes a brain question through grep instead of `brain_search`, we still get a measurement of whether the brain helped.

## [0.5.11] - 2026-06-27

### Added
- Eval worker now **auto-commits and pushes** `.brain/eval-log.jsonl` after every new row (both `gate_skipped` and full A/B verdicts). Previously rows accumulated in the local file until the user manually `git push`ed the brain repo, so the brein.sh telemetry page lagged reality by however many days. Uses the same inter-process file lock as `brain_update` so writes don't race. Silent on every failure — telemetry must never break the host.

## [0.5.10] - 2026-06-27

### Fixed
- `brein daemon launchd` now emits a **complete** plist. Previously it only wrote `BRAIN_MCP_TRANSPORT/HOST/PORT` — missing `BRAIN_REPO`, `BRAIN_RETRIEVAL_LOG`, `BRAIN_VECTOR_INDEX`, `BRAIN_EMBEDDING_MODEL`, and `BRAIN_EVAL_ENABLED`. Users who followed the install instructions got a daemon that silently failed every search because `BRAIN_REPO` defaulted to a non-existent path. Values are pulled from `~/.brein/config.json`.

## [0.5.9] - 2026-06-27

### Fixed
- **A/B eval actually runs now.** Previously `maybe_eval` was wired only to `brain_evidence` and ran the A/B in a `threading.Thread(daemon=True)` — which got killed the moment the stdio MCP process exited (i.e. always). The eval log hadn't gained a row since the last manual `brain-eval` run. Now:
  - Eval is wired into `brain_search` in addition to `brain_evidence`.
  - The worker is a detached subprocess (`brein eval tick` over `start_new_session=True`) so it survives the MCP server's exit, same pattern as the consistency checker.
  - Persistent dedup at `~/.brein/eval-seen.jsonl` skips queries already evaluated in the last `BRAIN_EVAL_DEDUP_HOURS` (default 24h).
  - **LLM gate before the A/B**: one cheap "is this query significant — yes/no" call decides whether to spend the full A/B budget. Routine lookups produce a `kind: "gate_skipped"` row; meaningful queries trigger a full A/B with a verdict row. Both cases land in `eval-log.jsonl` so you can see the gate working.

## [0.5.8] - 2026-06-27

### Security / Data integrity
- **Synchronous push.** `brain_update` now waits for `git push` and reports `pushed: "ok" | "failed"` with a `push_error` field on failure. The old daemon-thread `_bg_push` returned `pushed: "pending"` synchronously and silently swallowed remote rejections (pre-receive hook, divergence, network). Commits could accumulate locally with no signal — fixed.
- **Inter-process write lock.** Two `brain_update` calls from concurrent `brain-mcp` processes used to race the pull → write → commit → push sequence; both could return success while neither commit landed. Now serialized via `fcntl.flock` on `<REPO>/.git/brein-write.lock` covering the full sequence (including network push). The previous per-process `_push_lock` was a no-op across processes — each MCP stdio client spawns its own.
- **AWS access key scanner** (carried over from 0.5.7): `AKIA…` / `ASIA…` patterns now rejected by `_detect_secrets`.

### Fixed
- **Frontmatter parser/validator gaps surfaced by E2E suite:**
  - Tab-indented required fields (e.g. `\ttype: note`) previously passed existence-by-substring check but were invisible to the parser. Validator now requires fields at column 0 inside the frontmatter block.
  - Duplicate frontmatter keys (e.g. two `status:` lines) silently last-wins. Now rejected with `duplicate frontmatter key <key>`.
- **Append mode silent corruption.** `brain_update(mode="append")` now rejects content beginning with a `---` block followed by another `---`, which would create a phantom second frontmatter block the validator never sees.
- **Vector index shape crash.** A `vector-index.json` whose root is a JSON list (not a dict) used to crash `brain_search` with raw `'list' object has no attribute 'get'`. Now treated as corrupted and rebuilt, like other corruption modes.

### Added
- **103 end-to-end tests** under `tests/test_*.py` driving the real MCP server (no mocks) over real fastembed across 10 surfaces: concurrency, append mode, pull-FF / push failure, index corruption, HTTP streamable-http transport, frontmatter parser, path traversal & symlinks, `brain_evidence` & `brain_audit`, rerank, and external-edit / index freshness. New `test.yml` already runs these on every push and PR.

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
