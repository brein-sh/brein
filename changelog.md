# Changelog

All notable changes to brein are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

A push to `main` that adds a new `## [X.Y.Z] - YYYY-MM-DD` heading is auto-tagged `vX.Y.Zf` and published by `publish.yml`. Tags ending in `f` skip tests (force release).

## [0.5.30] - 2026-06-28

### Added
- **`brain_admitted_no_answer` boolean on every A/B eval row.** Structured signal from the judge LLM: true when answer A explicitly admits it could not find a useful answer in the brain ("I don't have this", "no record of", "couldn't find"), false when A merely answered poorly. Replaces the prior brittle regex-on-brain-answer in the telemetry UI. Old rows lack the field → falsy → not counted. No backfill.

## [0.5.29] - 2026-06-28

### Added
- **`evolve_recheck` — auto-rerun A/B on every loss the cycle examined.** After improvements commit, each loss question fires a fresh `_run_ab` tagged `trigger: "evolve_recheck:<evolve_id>"` and a fresh `query_hash`. This closes the feedback loop: you can now actually measure whether an evolve cycle worked. Before today the only signal was "9 docs got patched"; now you also see "of those 9 questions, 7 now win the A/B (was 0/9)." Rechecks fan out across the same `BRAIN_EVOLVE_PARALLELISM=8` thread pool — ~$8 of API cost per 13-loss cycle, finishes in ~2 min. brein.sh UI can group eval rows by `evolve_id` to render before/after trajectories. Rechecks measure "did the specific edit close the gap" (warm-cache, intentional); the rolling 100-row eval still measures long-term real-world performance.

### Fixed
- **`_commit_all_edits` runs even when this cycle's `improved` counter is 0.** Discovered when v0.5.27 was killed mid-cycle: the agent's `Edit` tool calls had already mutated 9 brain docs on disk, but the parent died before the commit step. The next v0.5.28 cycle correctly observed "docs already contain the refs" and reported `skipped` for all 13 losses — but gated commit on `improved > 0` and so left the rescue work uncommitted. Fix: always attempt the commit (it's a no-op when the working tree is clean), so any orphaned edits from a killed prior run get rescued automatically on the next cycle.

## [0.5.28] - 2026-06-28

### Changed
- **Evolve now fans losses across a thread pool** (default `BRAIN_EVOLVE_PARALLELISM=8`). v0.5.27 ran 13 losses sequentially at ~40-90s each — 9-20 min per cycle. Each loss is one independent `ask_llm` against a different canonical doc; nothing was actually serial except a `for` loop. Now 13 losses finish in ~max(90s) ≈ 1.5 min at the same total dollar cost. Combined `evolve:` commit + push still serializes at the end under the same write lock. If two parallel agents somehow race on the same canonical doc, last-write-wins at the file level; git status at commit time captures whatever survives.
- **`loss_end` progress rows now carry `summary`, `escalation_reason`, `canonical_path`.** v0.5.27 logged just `kind` and `elapsed_s`, so a string of `skipped` results was a mystery until cycle_end. Now `tail -f ~/.brein/evolve-progress.jsonl` shows WHY each loss skipped/escalated/improved as it happens. Same observability-is-not-data rule — write failures swallowed.

## [0.5.27] - 2026-06-28

### Added
- **`~/.brein/evolve-progress.jsonl` — per-loss cursor for in-flight evolve runs.** A real evolve cycle on 13 losses takes 30+ minutes, and v0.5.26 wrote nothing observable until the very end — making "where is the worker right now?" unanswerable except by `pgrep`-watching claude children and guessing. Now `run_evolve` appends `cycle_start`, `loss_start`, `loss_end`, `cycle_end` rows with index/total/question/elapsed_s/running_totals/cycle_id, so `tail -f ~/.brein/evolve-progress.jsonl` shows a live cursor. Safe to fail (progress = observability, not data); per-loss appends are atomic enough for tail-following.

## [0.5.26] - 2026-06-28

### Fixed
- **Evolve filtered eval-log rows by `kind == "ab_run"` — a field that does not exist on A/B rows.** Caught when Samuel actually ran `brein evolve run --limit 13` and got `losses_examined: 0` despite 13 known no-brain wins in the log. Real schema: A/B verdict rows carry the verdict at the top level with NO `kind` field; only `gate_skipped` rows have `kind`. Now `_count_ab_runs` and `_read_recent_losses` detect A/B rows by `verdict ∈ {brain_better, tie, no_brain_better}`. Tests rewritten to use the real on-disk shape rather than my made-up shape — the original v0.5.24 tests passed against a fiction. Every test in `test_evolve.py` now uses rows that would round-trip through the actual eval pipeline.

## [0.5.25] - 2026-06-28

### Fixed
- **`brein evolve run` died with `NameError: name 'json' is not defined`.** The new `_cmd_evolve` handler in v0.5.24 used `json.dumps(...)` to print the result, but `cli.py` had never imported `json` (the existing eval/consistency handlers happened not to need it). Caught the moment Samuel actually ran the shipped command. Test added that exercises `_cmd_evolve` end-to-end and asserts the JSON output reaches stdout.

## [0.5.24] - 2026-06-28

### Added
- **`/evolve` — self-improvement loop.** The first 102 A/B eval rows produced one single failure mode across **all 13/13** no-brain wins: the no-brain answer cited concrete file paths / line numbers / function names that the brain doc lacked. Brain wasn't losing on knowledge — it was losing on **specificity**. So `evolve.run_evolve(limit=50)` now reads the recent `no_brain_better` rows from `eval-log.jsonl`, and for each loss invokes the agentic LLM with `Read,Grep,Glob,Edit` tools to: (1) find the canonical brain doc for the question, (2) verify the concrete refs the no-brain answer used against the actual source under `~/Documents/GitHub/<repo>`, (3) edit the brain doc to add a `## Source references` section listing the verified refs. All edits land in one combined `evolve:` commit + push under the same write lock `brain_update` and consistency use. Per-run results go to `~/.brein/evolve-log.jsonl` with per-loss outcomes (`improved`/`skipped`/`escalated`). Auto-fires every `BRAIN_EVOLVE_EVERY=50` ab_runs (recursion-guarded via `BRAIN_EVOLVE_IN_PROGRESS`). Manual: `brein evolve run` / `brein evolve status`. New MCP tool: `brain_evolve_status`. Hardcoded "no edits without verified refs" rule — agent must Grep the source and confirm each path; never paste an unverified ref.

## [0.5.23] - 2026-06-28

### Changed
- **`brain_read` takes an optional `max_chars` arg; default lowered to 50k.** Yesterday's 0.5.22 shipped `brain_read` with a hardcoded 80k ceiling and no caller override — same paternalism class as the just-dropped `SIMILAR_THRESHOLD = 0.80`, two functions over. Now `max_chars: int | None = None` (None → use the env-configurable `BRAIN_MAX_READ_CHARS` default, now 50k), pass `0` for uncapped, pass a smaller number for a head. Response also includes `total_chars` so the caller can decide whether to re-call uncapped. Cap exists as a context-budget default, not a policy — agents that know their context can grow it.

## [0.5.22] - 2026-06-28

### Added
- **`brain_read` is now an actual MCP tool.** The eval data shows 13% of A/B losses are cases where the brain arm found the right doc via `brain_search` but never got the body — there was no exposed "give me the full doc at this path" tool. An internal `brain_read` helper existed for `brain_evidence` but was never registered as `@mcp.tool`, so agents fell back to vanilla `Read` (no path-safety, no telemetry, no `MAX_READ_CHARS` envelope). Now `brain_read(file_path)` is callable directly, returns the full body up to 80k chars, parses frontmatter for the caller, and logs a `kind: "read"` row so retrieval analytics see direct loads alongside search/evidence.

### Changed
- **`brain_evidence` no longer hard-truncates doc excerpts at 2500 chars.** Same eval-data motivation: 2500 chars silently chopped small canonical decision docs (often 4–6k) mid-sentence, so the answering agent got a fragment of the very thing it asked for. Raised the per-doc excerpt cap to 8000; single-doc bundles return the full body. Multi-doc bundles still get capped to protect context, but at a level where a typical decision doc survives intact.
- **`brain_search` now applies a deterministic post-rank boost for `source_of_truth: true` + recency.** Vector similarity alone can rank a narrative note equal-to-or-above the canonical doc that actually settles the topic — exactly the loss pattern dominating our internal/mixed no-brain wins. The new pass adds +0.05 for `source_of_truth: true` and up to +0.02 for fresh `last_reviewed`/`decided` dates (linear decay over 3 years). Small enough that a clearly-better vector hit still wins; large enough that ties go to the doc that arbitrates.
- **Dropped the `SIMILAR_THRESHOLD = 0.80` filter in the consistency checker.** Now that the post-write judge is agentic and costs ~$0.30/call, hardcoding "below 0.80 vec_score = don't bother judging" was paternalism — a real near-conflict at 0.78 was getting silently dropped. The agent already sees `vec_score=X.XXX` in each NEIGHBOR header and can decide "weak overlap → kind: ok" itself in one cheap turn. Saves zero meaningful money; removes a silent-miss class.

### Fixed
- **`publish.yml` no longer tries to `sed`-rewrite a pyproject.toml `version =` line that hasn't existed since v0.5.18.** Both v0.5.20 and v0.5.21 published locally but never made it to PyPI because the workflow's "Sync pyproject.toml version" step ran a sed that matched nothing (fine), then `grep -E '^version = '` which failed under `set -e` (since the project switched to `dynamic = ["version"]` reading from `changelog.md` via hatch). Dead step removed; version is sourced dynamically at build time and the auto-tag flow already guarantees the changelog heading is in the checkout.

## [0.5.21] - 2026-06-28

### Added
- **Regression tests for the silent-failure class that bit us 0.5.16 → 0.5.20** (`tests/test_daemon_env_silent_failures.py`). 13 tests that simulate the launchd-daemon environment our existing suite never exercised: restricted PATH, `sys.argv[0]` pointing at the server launcher, `shutil.which` returning None, full-path CLI resolution. Each maps directly to a bug we shipped today.

### Fixed
- **`cli == "claude"` equality check broke `--allowed-tools` when `_which_llm_cli` returned a full path** (e.g. `/opt/homebrew/bin/claude` from the known-paths probe added in 0.5.19). Replaced with `Path(cli).name == "claude"`. Caught by the new regression tests — in production the supersede still worked because claude defaults to having tools when the flag is absent, but the agentic flag was never actually being passed by daemon-spawned workers. Anyone restricting tools to specific names (security, cost) was getting the unrestricted default.

## [0.5.20] - 2026-06-28

### Changed
- **Agentic consistency prompt now tells the agent to verify against source code.** Previously the prompt only talked about reading other brain docs to break ties — but the agent already had `Read`/`Grep`/`Glob` access to any file the worker process could see, including local source checkouts under `~/Documents/GitHub/`. New evidence ladder: source code > `source_of_truth: true` frontmatter > newer date > snippet match. A doc that contradicts the code is the loser regardless of metadata. Common pmxt repo locations are listed in the prompt as a hint.

## [0.5.19] - 2026-06-28

### Fixed
- **`_which_llm_cli` now probes known install paths when `shutil.which` fails.** Same root cause as 0.5.18 (launchd's minimal PATH), one layer deeper: even after 0.5.18 fixed the worker spawn, the worker itself called `shutil.which("claude")` to pick the judge CLI — and that also returned None inside the daemon's env, so consistency findings landed as `judge: "stub"` / `escalation_reason: "judge_unavailable"` instead of running the agentic resolver. Now `_which_llm_cli` falls back to `/opt/homebrew/bin/`, `/usr/local/bin/`, `~/.local/bin/`, `~/.claude/local/` for `claude`/`codex`/`gemini` after PATH lookup misses. Found by triggering brain_update against the v0.5.18 daemon: spawn worked (real PID), worker ran end-to-end, but emitted a stub finding instead of an agent run.

## [0.5.18] - 2026-06-28

### Fixed
- **Consistency worker silently failed under launchd because `brein` wasn't on the daemon's PATH.** launchctl gives processes a minimal default PATH (`/usr/bin:/bin:/usr/sbin:/sbin`) that does NOT include `~/.local/bin` where the `brein` CLI lives. After 0.5.17 fixed `_brein_executable` to prefer `shutil.which("brein")`, that call now returned None inside the daemon, fell through to the literal string `"brein"`, and `subprocess.Popen` raised `FileNotFoundError` — caught by `server.py`'s try/except, surfacing as `consistency_check_pid: null`. Now we invoke `python -m brain_mcp.cli consistency check <path>` via `sys.executable`, exactly like `eval._spawn_eval_worker` does. PATH-independent, no plist changes needed. Found by running `brain_update` against the v0.5.17 daemon and getting `consistency_check_pid: null` with nothing in the worker log.

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
