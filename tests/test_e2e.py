"""End-to-end tests over the real MCP server.

Covers the load-bearing invariants:
- Round-trip: brain_search returns ranked hits for known seed content.
- Telemetry conservation: every successful search appends exactly one line
  to BRAIN_RETRIEVAL_LOG, and every line is well-formed JSON with the keys
  downstream eval depends on.
- Write loop: brain_update writes a file, commits it, and pushes to the
  configured remote.

No mocks. The server is launched as a real subprocess speaking JSON-RPC
over stdio, exactly as a real MCP client would.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from conftest import make_frontmatter

# Drive the importable brain_mcp module rather than `shutil.which("brain-mcp")`,
# so tests always exercise the working tree — not a stale globally-installed
# version. Same Python the test runner uses → editable install picked up.
BRAIN_MCP_CMD = sys.executable
BRAIN_MCP_ARGS = ["-m", "brain_mcp.server"]


async def _call(env: dict[str, str], tool: str, args: dict):
    """Return (text, is_error). MCP tools may surface ValueError/RuntimeError
    as `isError=True` with the exception text as content — we keep both
    so error-path tests can assert on either."""
    params = StdioServerParameters(command=BRAIN_MCP_CMD, args=BRAIN_MCP_ARGS, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            assert result.content, f"{tool} returned no content"
            return result.content[0].text, bool(getattr(result, "isError", False))


def _run(env, tool, args):
    """Tools that succeed always return JSON. Use this for happy-path tests."""
    text, is_error = asyncio.run(_call(env, tool, args))
    assert not is_error, f"unexpected tool error: {text}"
    return json.loads(text)


def _run_raw(env, tool, args):
    """Returns (parsed_or_text, is_error). For tests that expect errors."""
    text, is_error = asyncio.run(_call(env, tool, args))
    try:
        return json.loads(text), is_error
    except json.JSONDecodeError:
        return text, is_error


def _have_fastembed_model() -> bool:
    """True when the real semantic embedder is loadable. CI installs it via
    fastembed model cache. Locally it's present after one successful run."""
    try:
        from fastembed import TextEmbedding
        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return True
    except Exception:
        return False


needs_embedder = pytest.mark.skipif(
    not _have_fastembed_model(),
    reason="real fastembed model unavailable (offline / first run)",
)


def _log_lines(env) -> list[dict]:
    p = Path(env["BRAIN_RETRIEVAL_LOG"])
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_search_round_trip(brain_env):
    out = _run(brain_env, "brain_search", {"query": "quokka nasturtium"})
    assert out["status"] == "ready", out
    paths = [r["path"] for r in out["results"]]
    assert any("alpha" in p for p in paths), f"alpha.md missing from results: {paths}"


def test_telemetry_conservation(brain_env):
    """Every successful brain_search emits exactly two log lines:
    - one generic tool-call trace (from @logged on every MCP tool), and
    - one app-specific 'search' record with the hit paths.
    Both must always fire — if either disappears, downstream eval breaks
    silently.
    """
    def by_kind(rows):
        out = {}
        for r in rows:
            out.setdefault(r.get("kind"), []).append(r)
        return out

    before = by_kind(_log_lines(brain_env))
    n_search_before = len(before.get("search", []))
    n_call_before = len(before.get("tool_call", []))

    queries = ["quokka", "walrus", "pickle", "moonlight", "nasturtium"]
    for q in queries:
        out = _run(brain_env, "brain_search", {"query": q})
        assert out["status"] == "ready", f"search for {q!r} not ready: {out}"

    after = by_kind(_log_lines(brain_env))
    new_search = after.get("search", [])[n_search_before:]
    new_call = [r for r in after.get("tool_call", [])[n_call_before:]
                if r.get("gen_ai.tool.name") == "brain_search"]

    assert len(new_search) == len(queries), \
        f"expected {len(queries)} search rows, got {len(new_search)}"
    assert len(new_call) == len(queries), \
        f"expected {len(queries)} tool_call rows for brain_search, got {len(new_call)}"

    # Schema invariant on the freshly appended rows.
    for row in new_search:
        for k in ("ts", "kind", "question", "hits", "outcome"):
            assert k in row, f"search row missing {k!r}: {row}"
    for row in new_call:
        for k in ("ts", "kind", "gen_ai.tool.name", "latency_ms", "ok"):
            assert k in row, f"tool_call row missing {k!r}: {row}"
        assert row["ok"] is True


def test_write_loop(brain_env):
    """brain_update writes file → commits → pushes to bare remote."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/gamma.md"
    content = (
        make_frontmatter("Gamma penguin memo", ["bird"])
        + "Penguins prefer cold sardines.\n"
    )

    out = _run(brain_env, "brain_update", {
        "file_path": rel,
        "content": content,
        "commit_message": "test: add gamma penguin memo",
    })
    assert "error" not in out, out

    # File landed.
    assert (repo / rel).read_text() == content

    # Commit landed (subject line matches).
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--pretty=%s"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert "gamma penguin memo" in log, log

    # Push reached the bare remote.
    # Query refs/heads/main explicitly: bare repo HEAD may default to a
    # different branch on the runner (e.g. master), which would make a
    # bare `git log -1` fail with no such ref.
    bare_log = subprocess.run(
        ["git", "-C", str(repo.parent / "brain.git"), "log", "-1",
         "--pretty=%s", "refs/heads/main"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert "gamma penguin memo" in bare_log, bare_log


# ── Real semantic embedder path ──────────────────────────────────────────────

@needs_embedder
def test_semantic_recall_with_real_embeddings(brain_env):
    """Lexically-distant query must still recall the right doc when the
    fastembed backend is in use. With hash-fallback, this would fail —
    that's the point of the gate.
    """
    out = _run(brain_env, "brain_search", {"query": "what marsupial eats plants in Australia"})
    assert out["status"] == "ready", out
    backend = (out.get("vector") or {}).get("backend", "")
    assert backend.startswith("fastembed:"), \
        f"expected fastembed backend, got {backend!r} (hash-fallback masks real bugs)"
    top_paths = [r["path"] for r in out["results"][:2]]
    assert any("alpha" in p for p in top_paths), \
        f"alpha.md (quokka) should rank top-2 for marsupial query, got {top_paths}"


def test_write_then_search_finds_new_content(brain_env):
    """Write a new doc → searching for terms only in that doc must find it.
    Closes the write → reindex → search loop. If the index doesn't get
    rebuilt after a write, this fails.
    """
    rel = "docs/delta.md"
    body = (
        make_frontmatter("Delta capybara observation", ["rodent"])
        + "Capybaras congregate near hot springs and groom each other.\n"
    )
    out = _run(brain_env, "brain_update", {
        "file_path": rel,
        "content": body,
        "commit_message": "test: add capybara observation",
    })
    assert "error" not in out, out

    # Term ("capybara") appears in no other seed doc.
    found = _run(brain_env, "brain_search", {"query": "capybara"})
    assert found["status"] == "ready", found
    paths = [r["path"] for r in found["results"]]
    assert rel in paths, f"new doc not findable after write: {paths}"


# ── Error paths must surface, not silently succeed ───────────────────────────

def test_validator_rolls_back_bad_frontmatter(brain_env):
    """Doc missing required frontmatter must be rolled back: error returned,
    file not on disk, no new commit on either main or the bare remote.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/bad.md"
    head_before = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    out = _run(brain_env, "brain_update", {
        "file_path": rel,
        "content": "---\ntitle: only title, missing the rest\n---\n\nbody.\n",
        "commit_message": "test: should be rolled back",
    })
    assert out.get("rolled_back") is True, out
    assert "error" in out

    assert not (repo / rel).exists(), "rolled-back file should not be on disk"

    head_after = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert head_after == head_before, "HEAD must not advance on rollback"


def test_path_traversal_blocked(brain_env):
    """`file_path` must not escape the brain repo. The server should reject
    with an error — never write outside REPO_PATH.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    leak_target = repo.parent / "leaked.txt"

    out, is_error = _run_raw(brain_env, "brain_update", {
        "file_path": "../leaked.txt",
        "content": make_frontmatter("leak", ["bad"]) + "should not write\n",
        "commit_message": "should not commit",
    })
    # Either: structured {"error": ...} JSON, OR MCP tool error with the
    # ValueError text. Both are acceptable; silent success is not.
    msg = (out["error"] if isinstance(out, dict) and "error" in out
           else out if isinstance(out, str) else json.dumps(out))
    assert is_error or "error" in (out if isinstance(out, dict) else {}), \
        f"path traversal silently succeeded: {out}"
    assert "escape" in msg.lower() or "repo-relative" in msg.lower() or "outside" in msg.lower(), msg
    assert not leak_target.exists(), "path traversal must not write outside repo"


def test_secret_scanning_blocks_write(brain_env):
    """Content matching the secret-scanner must be refused. We use a
    bogus-but-pattern-matching AWS access key id."""
    rel = "docs/secrets-test.md"
    body = (
        make_frontmatter("secrets test", ["op"])
        + "Here's a key for the runbook: AKIAIOSFODNN7EXAMPLE\n"
    )
    out = _run(brain_env, "brain_update", {
        "file_path": rel,
        "content": body,
        "commit_message": "test: should be refused",
    })
    assert "error" in out, out
    assert "secret" in out["error"].lower(), out
    repo = Path(brain_env["BRAIN_REPO"])
    assert not (repo / rel).exists()


# ── CLI smoke ────────────────────────────────────────────────────────────────

def test_cli_doctor_exits_zero(brain_env):
    """`brein doctor` against a healthy world exits 0."""
    brein = [sys.executable, "-m", "brain_mcp.cli"]
    r = subprocess.run([*brein, "doctor"], env=brain_env, capture_output=True, text=True)
    assert r.returncode == 0, f"brein doctor exit={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"


def test_cli_index_status_ready(brain_env):
    """`brein index status` reports ready after conftest prebuild."""
    brein = [sys.executable, "-m", "brain_mcp.cli"]
    r = subprocess.run([*brein, "index", "status"], env=brain_env, capture_output=True, text=True)
    assert r.returncode == 0
    assert "status: ready" in r.stdout, r.stdout
