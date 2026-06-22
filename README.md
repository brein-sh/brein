# brain-mcp

MCP server for a brain — a git repo of markdown that an LLM reads and writes.

Exposes `brain_read`, `brain_search`, `brain_list`, `brain_update`, `brain_answer`, `brain_audit`, `brain_classify`, `brain_retrieval_log` over a local brain repo.

## Setup

```bash
git clone https://github.com/brainincorp/brain-mcp.git
cd brain-mcp
uv sync
```

Point an MCP-capable client (Claude Code, Cursor, Codex) at `examples/brain.mcp.json` — replace the two placeholder paths with absolute paths.

## Config

| Env | Default | Purpose |
|---|---|---|
| `BRAIN_REPO` | `~/.braincorp/brain` | Path to the brain repo (must be a git repo with `docs/`). |
| `BRAIN_RETRIEVAL_LOG` | `~/.braincorp/retrieval-log.jsonl` | Where every tool call is logged. |
| `BRAIN_VECTOR_INDEX` | `~/.braincorp/vector-index.json` | Vector index cache. |
| `BRAIN_TELEMETRY_FLUSH_EVERY` | `10` | Auto-commit + push the retrieval log every N events. |

## Continuous eval (on by default)

Every `brain_answer` call decides whether the query is worth comparing to a no-brain baseline. If yes, a background thread:

1. Re-asks the question with brain evidence (= "with-brain" answer)
2. Re-asks the question with no evidence (= "no-brain" answer)
3. LLM judge picks better/tie/worse
4. Appends one row to `.brain/eval-log.jsonl`

**Inference runs through whichever client CLI is on your PATH** — `claude`, `codex`, or `gemini` — so it uses your existing subscription auth. No extra API key needed. Falls back to OpenRouter only if no CLI is found (useful in CI / headless).

Triggers (free to detect):
- `dont_know` — answer text matched a "couldn't find / no record / not in" pattern
- `novel_hash` — first time this query has been seen this process

| Env | Default | Purpose |
|---|---|---|
| `BRAIN_EVAL_ENABLED` | `on` | Set to `off` to disable entirely. |
| `BRAIN_EVAL_CLIENT` | `claude,codex,gemini` | CLI preference order; first one on PATH wins. |
| `BRAIN_EVAL_CLI_TIMEOUT_S` | `120` | Per-CLI-call timeout. |
| `BRAIN_EVAL_OPENROUTER_KEY` | — | Fallback only — used when no CLI is on PATH. |
| `BRAIN_EVAL_MODEL` | `deepseek/deepseek-v4-flash` | Model used with the OpenRouter fallback. |
| `BRAIN_EVAL_TIMEOUT_S` | `60` | OpenRouter request timeout. |

A recursion guard env var (`BRAIN_EVAL_IN_PROGRESS=1`) is set on subprocessed CLIs so the child's `brain_answer` calls don't loop the eval back into themselves.

The dataset (`.brain/eval-log.jsonl`) grows over time and lives inside your brain repo — searchable by `brain_search`, committable by the same auto-commit loop, viewable by anything that reads the file. Run aggregates on it to produce a with/without effectiveness matrix any time you want.

Failures are silently swallowed — eval must never break a brain call.

## Layout

```
src/brain_mcp/   ← the package
scripts/         ← launcher + eval_retrieval.py
examples/        ← brain.mcp.json + eval-cases.example.json
```

## License

MIT.
