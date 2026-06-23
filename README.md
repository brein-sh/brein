<p align="center">
  <a href="https://brein.sh"><img src="assets/brain.gif" alt="brein" width="320"></a>
</p>

<h1 align="center">brein</h1>

<p align="center"><em>a brain that lives in your company.</em></p>

brein is a local MCP server for a company memory that lives in git: markdown docs in your repo, readable and writable by AI agents. Works with Claude Code, Cursor, Codex, Hermes, and any stdio MCP client.

Website: https://brein.sh

## Quickstart

```bash
git clone https://github.com/brein-sh/brein && cd brein
uv sync
brein setup          # interactive: brain repo, paths, eval, MCP client
brein mcp claude     # prints the snippet to paste into your client config
```

Then restart your MCP client. Verify with `brein doctor`.

Requirements: Python 3.11+, `uv`, `git`, an MCP-capable client.

## CLI

```bash
brein setup [section]   # all sections, or one of: repo paths vector eval mcp
brein doctor            # ok/warn/fail health checks
brein mcp <client>      # claude | cursor | codex | generic
brein config            # show resolved config
brain-eval --limit 20   # local retrieval evals against $BRAIN_REPO
```

## Tools exposed over MCP

| Tool | Purpose |
| --- | --- |
| `brain_search(query, mode="hybrid")` | Keyword / vector / hybrid retrieval with optional rerank. |
| `brain_evidence(question)` | Ranked docs + snippets + excerpts + citations in one round-trip. The client agent writes the final answer. |
| `brain_read(file_path)` | Repo-relative read with frontmatter. Absolute paths and `.git` are blocked. |
| `brain_list(directory="docs")` | List markdown under a repo-relative dir. |
| `brain_update(file_path, content, commit_message, mode)` | Write to an allowed path, validate, commit, push. |
| `brain_audit()` | Repo cleanliness, doc/frontmatter counts, log + index health. |
| `brain_retrieval_log(...)` | Telemetry; search/read also auto-log. |

Write policy: paths under `docs/`, `skills/`, `templates/`, plus `AGENTS.md` / `README.md` / `CONTRIBUTING.md` at the root. Secrets are pattern-blocked.

## Configuration

`brein setup` writes `~/.brein/config.json`. Environment variables override the file and are documented in [`src/brain_mcp/config.py`](src/brain_mcp/config.py). The big ones:

- `BRAIN_REPO` — path to your brain git repo (required)
- `BRAIN_RETRIEVAL_LOG` — JSONL telemetry path
- `BRAIN_VECTOR_INDEX` — vector index cache path
- `BRAIN_EMBEDDING_MODEL` — fastembed model (default `BAAI/bge-small-en-v1.5`)

## Project layout

```text
src/brain_mcp/          MCP server + `brein` CLI
src/brain_mcp/_scripts/ bundled helpers (eval, index, validate)
```

## Contributing

Issues and PRs welcome. See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
