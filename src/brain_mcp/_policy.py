"""Agent policy template — the read-at-start + write-on-triggers behavior
that makes brein actually get used by agents."""


POLICY_TEMPLATE = """# brein (MCP: brain)

The brain is a git-backed knowledge repository at:

    {repo_path}

Read it with your normal file tools (Read, Glob, Grep, etc.) — it's a regular
directory. Use the brein MCP tools only for the things plain file tools
can't do: semantic search and policy-gated writes.

## Tools

- `brain_search` — semantic-only retrieval (embeddings). Use for paraphrased
  or conceptual queries. **Returns a status payload (not results) if the
  vector index isn't ready** — see "When brain_search returns status" below.
- `brain_evidence` — one-shot ranked-docs-plus-citations bundle for grounded
  question answering. Calls search + reads top hits in one round-trip.
- `brain_index_status` — inspect or kick the background index builder.
  Use when brain_search keeps returning non-ready and you want progress
  or want to force a restart (`restart_if_stalled=True`).
- `brain_update` — REQUIRED for all writes. Enforces secret blocking,
  allowed-path policy, atomic validate → commit → push. Never write to the
  brain repo via Edit / Write / shell — go through `brain_update`.
- `brain_audit` — repo health (cleanliness, doc counts, log/index status).

## When `brain_search` returns status (not results)

`brain_search` is embeddings-only. If the index isn't ready, it returns:

    {{
      "status": "building" | "stalled" | "missing" | "empty",
      "action": "use_grep",
      "hint": "...",
      "repo_path": "{repo_path}",
      "progress": "448/2167 (20%)",
      "auto_spawned_worker": true
    }}

When you see this, do this:

1. **Use your normal Grep / Read / Glob tools** over `{repo_path}/docs` —
   the brain is just a directory of markdown, you don't need brain_search
   to read it. Grep for keywords, read the files you find.
2. **Retry `brain_search` later** in the same conversation. The worker
   auto-spawns on missing/stalled, so by the next round-trip it may be
   `ready` (or further along).
3. If status is `stalled` and you want a fresh start, call
   `brain_index_status(restart_if_stalled=True)`.
4. **Never treat status=building as "no results exist"** — the docs are
   right there in the repo, just not semantically indexed yet.

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

### Write Often
Default to writing. If a turn taught you anything not already in the brain —
a name, a path, a decision, a relationship, a status — `brain_update` it
before stopping. The brain is a separate store from your context window:
facts in CLAUDE.md / auto-memory / tool results are NOT in the brain until
you write them.

Don't ask whether to write. Don't wait for the user to confirm. Don't defer
"until the repo list is stable" or "until you have more context". A short,
honest entry now beats a perfect entry never. You can always update it next
turn.

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
