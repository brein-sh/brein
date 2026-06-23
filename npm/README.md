# brein (npm)

Thin Node wrapper for [brein](https://github.com/brein-sh/brein) — the MCP server for a company memory that lives in git.

## Quickstart

```bash
npx brein init
```

Verifies Python 3.11+ and [`uv`](https://docs.astral.sh/uv/), installs the brein Python package via `uv tool install`, then runs the setup wizard.

After install, the wrapper forwards subcommands to the real CLI:

```bash
npx brein doctor
npx brein mcp claude
npx brein setup mcp
```

Or use the installed binary directly (`brein`, `brain-mcp`, `brain-eval`).

## Requirements

- Node 18+
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (single static binary, no Python prereq for install)
- `git`

The wrapper prints actionable install hints if anything is missing.

## License

MIT.
