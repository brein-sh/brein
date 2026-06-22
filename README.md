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

## Continuous eval (optional)

When enabled, every `brain_answer` call decides whether the query is worth comparing to a no-brain baseline. If yes, a background thread:

1. Re-asks the question with brain evidence (= "with-brain" answer)
2. Re-asks the question with no evidence (= "no-brain" answer)
3. Cheap LLM judge picks better/tie/worse
4. Appends one row to `.brain/eval-log.jsonl`

Triggers (free to detect):
- `dont_know` — answer text matched a "couldn't find / no record / not in the repo" pattern
- `novel_hash` — first time this query has been seen this process

Set these env vars:

| Env | Default | Purpose |
|---|---|---|
| `BRAIN_EVAL_ENABLED` | `off` | Set to `on` to turn the loop on. |
| `BRAIN_EVAL_OPENROUTER_KEY` | — | OpenRouter API key for the A/B + judge calls. Required. |
| `BRAIN_EVAL_MODEL` | `deepseek/deepseek-v4-flash` | Model used for both with-brain and no-brain answers. |
| `BRAIN_EVAL_JUDGE_MODEL` | `deepseek/deepseek-v4-flash` | Model used to pick the winner. |
| `BRAIN_EVAL_TIMEOUT_S` | `60` | Per-request timeout. |

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
