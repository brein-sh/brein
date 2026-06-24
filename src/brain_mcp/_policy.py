"""Agent policy template — the read-at-start + write-on-triggers behavior
that makes brein actually get used by agents."""


POLICY_TEMPLATE = """# brein (MCP: brain)

The brain is a git-backed knowledge repository at:

    {repo_path}

Read it with your normal file tools (Read, Glob, Grep, etc.) — it's a regular
directory. Use the brein MCP tools only for the things plain file tools
can't do: semantic search and policy-gated writes.

## Tools

- `brain_search` — semantic / hybrid retrieval. Use when grep would miss
  paraphrases or conceptual queries. For exact-string lookups, just grep.
- `brain_evidence` — one-shot ranked-docs-plus-citations bundle for grounded
  question answering. Calls search + reads top hits in one round-trip.
- `brain_update` — REQUIRED for all writes. Enforces secret blocking,
  allowed-path policy, atomic validate → commit → push. Never write to the
  brain repo via Edit / Write / shell — go through `brain_update`.
- `brain_audit` — repo health (cleanliness, doc counts, log/index status).

## Session Start — Read the Brain

At the beginning of every conversation, before doing any work:

1. Read `{repo_path}/docs/index.md` (or `docs/start-here.md` if it exists) to load current priorities and context.
2. If the request relates to a specific area (projects, contacts, decisions, knowledge), navigate to the relevant subdirectory and read the docs you need. Use `brain_search` if the right doc isn't obvious from filenames.

## During the Session — Write to the Brain

Whenever you learn something useful across future sessions, write it via `brain_update`. Examples:

- A decision was made (architecture, product, business) → `docs/decisions/`
- New contact or company info → `docs/contacts/` or `docs/companies/`
- Project status change or new project → `docs/projects/`
- Useful knowledge (API quirks, vendor info, research findings) → `docs/knowledge/`
- A new skill or workflow → `docs/skills/`

Follow the existing file structure and naming conventions in the brain.

## Mandatory Write Triggers

You MUST write to the brain in these situations — no exceptions:

### Before Context Compaction
Write a session summary to `docs/knowledge/session-learnings/<YYYY-MM-DD>-<topic>.md`: what was learned, decisions made, mistakes and their fixes, current project state if it changed.

### After Repeated Mistakes
If you make the same mistake **twice**, immediately write to `docs/knowledge/pitfalls/`: what the mistake was, why it kept happening, the correct approach.

### After Debugging Breakthroughs
When a non-obvious bug is solved, write the root cause and fix to `docs/knowledge/debugging/`.

### After Architecture or Design Decisions
Capture confirmed approaches and rejected alternatives in `docs/decisions/`.

## Rules

- Do NOT ask "should I write this to the brain?" — just do it when it's clearly useful.
- DO mention what you wrote so the user is aware (e.g., "Saved the decision to docs/decisions/...").
- Do NOT write ephemeral or conversation-specific details — only durable knowledge.
- Do NOT write secrets, credentials, raw private chats, or uncurated personal facts.
- When in doubt about where, read `docs/index.md` for the taxonomy.
- Writing to the brain is NOT optional — it is a core responsibility every session.
"""


def render(repo_path: str) -> str:
    return POLICY_TEMPLATE.format(repo_path=repo_path)
