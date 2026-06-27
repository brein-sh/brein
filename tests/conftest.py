"""E2E fixtures: throwaway $HOME, git-backed brain repo with bare remote.

Every test gets an isolated world. No mocks, no contamination of the user's
real ~/.brein.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Always drive the importable working tree, not a stale global install.
BRAIN_MCP_CMD = sys.executable
BRAIN_MCP_ARGS = ["-m", "brain_mcp.server"]
BREIN_CLI = [sys.executable, "-m", "brain_mcp.cli"]

# Distinctive terms chosen so a search match is unambiguous.
def _frontmatter(title: str, tags: list[str]) -> str:
    # Mirrors REQUIRED_DOC_PATTERNS in _scripts/validate_docs.py so brain_update
    # doesn't roll back our test writes for "missing field" reasons.
    return (
        "---\n"
        f"title: {title}\n"
        "owner: tests\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        f"tags: {tags}\n"
        "type: note\n"
        "---\n\n"
    )


SEED_DOCS = {
    "docs/alpha.md": _frontmatter("Alpha quokka note", ["marsupial"])
    + "The quokka eats nasturtium leaves on Tuesdays.\n",
    "docs/beta.md": _frontmatter("Beta walrus dispatch", ["pinniped"])
    + "The walrus contemplates pickles by moonlight.\n",
}


# Exposed for test_e2e.py so write-loop content stays in sync with the
# validator's required frontmatter.
def make_frontmatter(title: str, tags: list[str]) -> str:
    return _frontmatter(title, tags)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def brain_env(tmp_path: Path) -> dict[str, str]:
    """Build an isolated brain world + return the env dict to launch brain-mcp."""
    repo = tmp_path / "brain"
    bare = tmp_path / "brain.git"
    home = tmp_path / "home"
    (home / ".brein").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)

    for rel, body in SEED_DOCS.items():
        (repo / rel).write_text(body)

    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    # `-b main` matters on Ubuntu CI: without it the bare HEAD defaults to
    # `master`, so `git clone bare` produces an empty checkout (no docs/),
    # which breaks tests that clone the remote.
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(bare)], check=True)
    _git(repo, "config", "user.email", "test@brein.sh")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "push", "-q", "-u", "origin", "main")

    env = {
        **os.environ,
        "HOME": str(home),
        "BRAIN_REPO": str(repo),
        "BRAIN_RETRIEVAL_LOG": str(home / ".brein" / "retrieval-log.jsonl"),
        "BRAIN_VECTOR_INDEX": str(home / ".brein" / "vector-index.json"),
        "BRAIN_EVAL_ENABLED": "1",
    }

    # `brein doctor` reads ~/.brein/config.json (env vars alone aren't enough
    # for the file-based config check). Mirror the env values into the file.
    (home / ".brein" / "config.json").write_text(json.dumps({
        "repo_path": env["BRAIN_REPO"],
        "retrieval_log": env["BRAIN_RETRIEVAL_LOG"],
        "vector_index": env["BRAIN_VECTOR_INDEX"],
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "eval_enabled": True,
        "eval_host_order": ["claude", "codex", "gemini"],
    }, indent=2))

    # Pre-build the index synchronously so the first search returns 'ready',
    # not 'building'. Otherwise telemetry assertions fail (the gate path
    # short-circuits before _append_retrieval_log).
    # Drive the importable module so tests run against the working tree,
    # not a stale globally-installed brein.
    subprocess.run(
        [sys.executable, "-m", "brain_mcp.cli", "index", "build"],
        env=env, check=True, capture_output=True,
    )

    return env


# ── Shared MCP helpers ───────────────────────────────────────────────────────

async def _call_async(env, tool: str, args: dict):
    """Return (text, is_error) from one tool call over a fresh stdio session."""
    params = StdioServerParameters(command=BRAIN_MCP_CMD, args=BRAIN_MCP_ARGS, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            assert result.content, f"{tool} returned no content"
            return result.content[0].text, bool(getattr(result, "isError", False))


def call_tool(env, tool: str, args: dict) -> tuple[str, bool]:
    """Sync wrapper around one MCP tool call. Returns (text, is_error)."""
    return asyncio.run(_call_async(env, tool, args))


def run(env, tool: str, args: dict):
    """Happy-path helper. Asserts non-error, returns parsed JSON."""
    text, is_error = call_tool(env, tool, args)
    assert not is_error, f"unexpected tool error: {text}"
    return json.loads(text)


def run_raw(env, tool: str, args: dict):
    """Returns (parsed_or_text, is_error). Use this when the tool may error."""
    text, is_error = call_tool(env, tool, args)
    try:
        return json.loads(text), is_error
    except json.JSONDecodeError:
        return text, is_error


def have_fastembed_model() -> bool:
    """True when the real semantic embedder is loadable."""
    try:
        from fastembed import TextEmbedding
        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return True
    except Exception:
        return False


needs_embedder = pytest.mark.skipif(
    not have_fastembed_model(),
    reason="real fastembed model unavailable (offline / first run)",
)
