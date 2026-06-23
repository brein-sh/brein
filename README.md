<p align="center">
  <a href="https://brein.sh"><img src="https://brein.sh/icon.png" alt="brein logo" width="96" height="96"></a>
</p>

<h1 align="center">brein</h1>

<p align="center"><em>a brain that lives in your company.</em></p>

brein is a local MCP server for a company memory that lives in git: markdown docs in your repo, readable and writable by AI agents.

Your agents read it before they act. They write to it after they decide. The result is a practical, auditable knowledge layer for Claude Code, Cursor, Codex, Hermes, and any MCP-capable client.

Website: https://brein.sh

## What is in this repo?

This repo currently ships the Python MCP server package named `brain-mcp`. It exposes tools over stdio for:

- finding docs with hybrid keyword + vector search
- reading curated markdown with frontmatter
- returning evidence bundles with citations
- creating or updating safe paths in a target brain repo
- writing retrieval telemetry and continuous eval logs
- auditing brain repo health
- classifying whether text belongs in a company brain, personal memory, assistant memory, or nowhere

The public product/CLI flow shown on the website (`npx brein init`) is the intended onboarding direction, but it is not implemented in this package yet. Track setup/doctor/CLI work in issue #5: https://github.com/brein-sh/brein/issues/5

## Requirements

- Python 3.11+
- `uv` for local development and MCP launcher commands
- `git`
- a target brain repo that is itself a git repo
- an MCP client such as Claude Desktop/Code, Cursor, Codex, Hermes, or another client that supports stdio MCP servers

Optional:

- `fastembed` is installed by this package and provides real local embeddings. If model loading/download fails, brein falls back to deterministic hash vectors with degraded semantic recall.
- `claude`, `codex`, or `gemini` on PATH enables continuous with-brain/no-brain evals using your existing CLI auth.
- OpenRouter can be used as a headless eval fallback via `BRAIN_EVAL_OPENROUTER_KEY`.

## Install from source

```bash
git clone https://github.com/brein-sh/brein.git
cd brein
uv sync
uv run brain-mcp --help
```

After `uv sync` (or `pip install -e .`), `brein`, `brain-mcp`, and `brain-eval` are on PATH. Run `brein setup` to configure interactively, then `brein doctor` to verify.

## Create a brain repo

`BRAIN_REPO` should point at a git repo containing your team-shareable markdown. A minimal local repo looks like this:

```bash
mkdir -p ~/.brein/brain/docs
cd ~/.brein/brain
git init
git branch -M main
cat > docs/index.md <<'EOF'
---
title: Brain index
status: active
source_of_truth: true
tags: [index]
---

# Brain index

Start here. Add runbooks, decisions, customers, product notes, and operating docs under docs/.
EOF
git add docs/index.md
git commit -m "init brain"
```

`brain_update` and `brain_audit` regenerate `docs/index.md` and validate frontmatter using scripts bundled with the brein package — no per-target-repo setup is required. Override with `BRAIN_INDEX_SCRIPT` or `BRAIN_VALIDATE_SCRIPT` if you want to point at custom versions in your own repo.

## MCP client config

Generate a ready-to-paste snippet from your `brein setup` config:

```bash
brein mcp claude    # or cursor, codex, generic
```

Paste the printed `mcpServers` block into your client's MCP config. The same JSON shape works anywhere a stdio MCP server is accepted.

After restarting your client, ask it to list MCP tools. You should see:

- `brain_list`
- `brain_search`
- `brain_read`
- `brain_evidence`
- `brain_update`
- `brain_audit`
- `brain_retrieval_log`

## How to use the tools

Typical agent policy:

1. Use `brain_search` or `brain_evidence` before answering questions about company state.
2. Use `brain_read` for the exact files you cite.
3. Use `brain_update` after a durable decision, runbook change, or source-of-truth correction.
4. Do not write secrets, credentials, raw private chats, or uncurated personal facts.

Tool behavior summary:

| Tool | Purpose |
| --- | --- |
| `brain_search(query, mode="hybrid")` | Find markdown docs using keyword, vector, or hybrid retrieval. Supports `domain`, `tag`, `status`, optional reranking, and vector index rebuild. |
| `brain_evidence(question)` | Returns an evidence bundle (ranked docs + snippets + first-2.5k-chars excerpts + citations) for one round-trip grounded retrieval. Does NOT synthesize a final natural-language answer — the client agent writes that from the evidence and cites paths. |
| `brain_read(file_path)` | Reads a repo-relative file, returns frontmatter and content. Absolute paths and `.git` access are blocked. |
| `brain_list(directory="docs")` | Lists markdown files under a repo-relative directory. |
| `brain_update(file_path, content, commit_message, mode)` | Creates/replaces/appends an allowed file, validates the target repo, commits, and pushes to `origin main` in the background. |
| `brain_audit()` | Reports repo cleanliness, docs/frontmatter counts, retrieval log path, and vector index health. |
| `brain_retrieval_log(...)` | Manually appends retrieval outcome telemetry; search/read/answer also auto-log. |

## CLI

```bash
brein setup              # interactive wizard (all sections)
brein setup mcp          # re-run one section
brein doctor             # ok/warn/fail health checks
brein mcp claude         # print MCP snippet for a client
brein config             # show resolved config
```

## Commands

Development:

```bash
uv sync
uv run brain-mcp
```

Run the server self-test/audit against a target brain repo:

```bash
BRAIN_REPO=/path/to/brain BRAIN_MCP_SELF_TEST=1 uv run brain-mcp
```

Run local retrieval evals against your target brain repo:

```bash
BRAIN_REPO=/path/to/brain brain-eval --limit 20
BRAIN_REPO=/path/to/brain brain-eval --modes keyword vector hybrid hybrid_rerank --rerank-method heuristic
```

Compile sanity check:

```bash
python3 -m compileall src
```

## Configuration

Core environment variables:

| Env | Default in package | Purpose |
| --- | --- | --- |
| `BRAIN_REPO` | `/opt/data/repos/brain` | Target brain repo. Must be a git repo. Override via `brein setup` or env. |
| `BRAIN_MAX_READ_CHARS` | `80000` | Max content returned by `brain_read`. |
| `BRAIN_RETRIEVAL_LOG` | `/opt/data/brain-mcp/retrieval-log.jsonl` | JSONL retrieval/tool telemetry path. If inside `BRAIN_REPO`, telemetry flush can commit it. |
| `BRAIN_TELEMETRY_FLUSH_EVERY` | `10` | Auto-commit/push telemetry every N events when the log is inside the brain repo. |

Retrieval/vector variables:

| Env | Default | Purpose |
| --- | --- | --- |
| `BRAIN_VECTOR_INDEX` | `/opt/data/brain-mcp/vector-index.json` | Incremental vector index cache. |
| `BRAIN_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name. |
| `BRAIN_VECTOR_CHUNK_CHARS` | `1400` | Chunk size for vector indexing. |
| `BRAIN_VECTOR_CHUNK_OVERLAP` | `250` | Character overlap between chunks. |
| `BRAIN_EMBED_BATCH_SIZE` | `8` | Batch size for embedding writes. |
| `BRAIN_HASH_EMBED_DIMS` | `384` | Dimensions for degraded hash fallback vectors. |
| `BRAIN_HYBRID_KEYWORD_WEIGHT` | `0.55` | Keyword weight in hybrid scoring. |
| `BRAIN_HYBRID_VECTOR_WEIGHT` | `0.45` | Vector weight in hybrid scoring. |

Reranking variables:

| Env | Default | Purpose |
| --- | --- | --- |
| `BRAIN_RERANK_MAX_TOP_K` | `25` | Max candidates reranked. |
| `BRAIN_RERANK_TIMEOUT_SECONDS` | `25` | LLM rerank command timeout. |
| `BRAIN_RERANK_SNIPPET_CHARS` | `240` | Snippet chars included per candidate. |
| `BRAIN_RERANK_SNIPPET_COUNT` | `2` | Snippets included per candidate. |
| `BRAIN_RERANK_PROVIDER` | `openai-codex` | Provider label passed to Hermes rerank command when available. |
| `BRAIN_RERANK_MODEL` | `gpt-5.4-mini` for `openai-codex`, otherwise empty | Model label passed to Hermes rerank command when available. |
| `BRAIN_RERANK_COMMAND` | empty | Full custom rerank command. Overrides Hermes lookup. |
| `BRAIN_RERANK_BIN` | empty | Hermes binary override for LLM rerank. |

Continuous eval variables:

| Env | Default | Purpose |
| --- | --- | --- |
| `BRAIN_EVAL_ENABLED` | `on` | Set to `off` to disable continuous evals. |
| `BRAIN_EVAL_CLIENT` | `claude,codex,gemini` | CLI preference order for eval inference. |
| `BRAIN_EVAL_CLI_TIMEOUT_S` | `120` | Per-CLI eval timeout. |
| `BRAIN_EVAL_OPENROUTER_KEY` | empty | OpenRouter fallback key when no CLI is found. |
| `BRAIN_EVAL_MODEL` | `deepseek/deepseek-v4-flash` | OpenRouter fallback model. |
| `BRAIN_EVAL_TIMEOUT_S` | `60` | OpenRouter request timeout. |
| `BRAIN_EVAL_IN_PROGRESS` | unset | Internal recursion guard set on child CLI calls. |

## Safety and privacy model

brein is designed for team-shareable company knowledge, not arbitrary private memory.

Current safeguards:

- local stdio MCP server; no hosted brein service is required for core retrieval
- brain data lives in your git repo
- writes are restricted to `docs/`, `skills/`, `templates/`, `AGENTS.md`, `README.md`, and `CONTRIBUTING.md`
- repo-relative path enforcement blocks absolute paths, traversal, and `.git` reads
- write calls reject common secret patterns such as API keys, GitHub tokens, private keys, passwords, and seed phrases
- telemetry argument redaction masks sensitive-looking field names and truncates long strings
- eval failures are swallowed so eval never breaks a normal brain call

Important caveats:

- If `brain_update` succeeds, it commits and pushes to `origin main` by design.
- If `BRAIN_RETRIEVAL_LOG` is inside your brain repo, telemetry may be committed and pushed every `BRAIN_TELEMETRY_FLUSH_EVERY` events.
- Continuous eval may call local AI CLIs or OpenRouter if configured. Disable with `BRAIN_EVAL_ENABLED=off` for sensitive deployments.
- Secret detection is a guardrail, not a proof. Do not point brein at repos full of credentials or raw private exports.
- The default package paths under `/opt/data/...` reflect the original internal deployment and should usually be overridden for public/local use.

## Evals and retrieval quality

`brain_evidence` triggers a non-blocking A/B eval when enabled and an inference backend is available. It compares a with-brain answer to a no-brain answer, judges which is better, and appends JSONL rows to:

```text
$BRAIN_REPO/.brain/eval-log.jsonl
```

The manual retrieval eval script scores expected docs for local eval cases:

```bash
BRAIN_REPO=/path/to/brain brain-eval --limit 20
```

The eval runner falls back to a small built-in case set if `evals/brain_retrieval_eval_cases.json` isn't present.

## Project layout

```text
src/brain_mcp/                MCP server package
src/brain_mcp/cli.py          `brein` CLI entrypoint
src/brain_mcp/_scripts/       bundled scripts (eval, index, validate)
```

## Contributing

Issues and PRs are welcome. Good first areas:

- public setup/doctor/CLI flow: https://github.com/brein-sh/brein/issues/5
- cross-brain memory routing as a separate layer: https://github.com/brein-sh/brein/issues/7

Please keep README changes practical and implementation-accurate: if a command is aspirational, label it as roadmap.

## License

MIT. See `LICENSE`.
