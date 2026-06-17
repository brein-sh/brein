# brain-mcp

MCP server for a BrainIncorp brain — a git repo of markdown that an LLM reads and writes.

Exposes `brain_read`, `brain_write`, `brain_list`, `brain_search` over a local brain repo.

## Setup

```bash
git clone https://github.com/brainincorp/brain-mcp.git
cd brain-mcp
uv sync
```

Point an MCP-capable client (Claude Code, Cursor, etc.) at `brain.mcp.json` — replace the two placeholder paths.

## Config

| Env | Default |
|---|---|
| `BRAIN_REPO` | `~/.braincorp/brain` |
| `BRAIN_RETRIEVAL_LOG` | `~/.braincorp/retrieval-log.jsonl` |
| `BRAIN_VECTOR_INDEX` | `~/.braincorp/vector-index.json` |

The brain repo must be a git repo with a `docs/` directory.
