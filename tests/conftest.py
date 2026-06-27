"""E2E fixtures: throwaway $HOME, git-backed brain repo with bare remote.

Every test gets an isolated world. No mocks, no contamination of the user's
real ~/.brein.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
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
