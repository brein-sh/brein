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
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from conftest import make_frontmatter

BRAIN_MCP = shutil.which("brain-mcp") or "brain-mcp"


async def _call(env: dict[str, str], tool: str, args: dict) -> str:
    params = StdioServerParameters(command=BRAIN_MCP, args=[], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            assert result.content, f"{tool} returned no content"
            return result.content[0].text


def _run(env, tool, args):
    return json.loads(asyncio.run(_call(env, tool, args)))


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
    bare_log = subprocess.run(
        ["git", "-C", str(repo.parent / "brain.git"), "log", "-1", "--pretty=%s"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert "gamma penguin memo" in bare_log, bare_log
