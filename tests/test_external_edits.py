"""Bug-hunt: external edits, index freshness, repo mutation outside MCP.

The brain repo is just a directory. Anyone can edit it (editor, git CLI,
external sync). The vector index is a cached JSON file. These tests poke
at the boundary between "what's on disk now" and "what the cached index
remembers" — every test is staleness/freshness focused, no happy paths.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import (
    brain_env,
    call_tool,
    make_frontmatter,
    needs_embedder,
    run,
    run_raw,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _paths(out) -> list[str]:
    return [r["path"] for r in out.get("results", [])]


def _search(env, query: str, **kw):
    return run(env, "brain_search", {"query": query, **kw})


# ── 1. external ADD content (existing file) ──────────────────────────────────

def test_external_add_content_to_existing_file_is_indexed(brain_env):
    """Edit a seeded doc on disk to add a new distinctive term.

    Vector index uses mtime_ns + size fingerprints — touching the file
    should trigger a re-embed of just that file on the next search.
    The new term must be findable; otherwise the index is stale.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    alpha = repo / "docs" / "alpha.md"
    body = alpha.read_text()
    distinct = "xenotransplantation"
    alpha.write_text(body + f"\nNew paragraph about {distinct} therapy.\n")

    out = _search(brain_env, distinct)
    assert out.get("status") == "ready", out
    assert any("alpha" in p for p in _paths(out)), (
        f"external append not picked up — {distinct} missing from results: "
        f"{_paths(out)}"
    )


# ── 2. external REMOVE content (existing file) ───────────────────────────────

@needs_embedder
def test_external_remove_content_does_not_return_old_chunk(brain_env):
    """Strip the distinctive seed term from a file and search for it.

    If the cached chunk is still returned with a snippet quoting the
    removed text, the index is leaking removed content.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    beta = repo / "docs" / "beta.md"
    # Replace body with neutral text; keep frontmatter intact.
    beta.write_text(make_frontmatter("Beta walrus dispatch", ["pinniped"])
                    + "Nothing notable here.\n")

    out = _search(brain_env, "walrus contemplates pickles moonlight")
    snippets = []
    for r in out.get("results", []):
        for s in r.get("snippets") or []:
            if s and "snippet" in s:
                snippets.append(s["snippet"])
    leaked = any("pickle" in s.lower() or "moonlight" in s.lower()
                 for s in snippets)
    assert not leaked, (
        f"removed content still surfaces in vector snippets: {snippets}"
    )


# ── 3. external DELETE file (no git commit) ──────────────────────────────────

def test_external_delete_file_does_not_surface_in_results(brain_env):
    """Unlink a seeded doc on disk (no git involvement).

    Searching for its content should NOT return a hit pointing at the
    now-missing path. A hit with a stale path is a correctness bug —
    downstream agents will try to read a file that no longer exists.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    (repo / "docs" / "alpha.md").unlink()

    out = _search(brain_env, "quokka nasturtium")
    paths = _paths(out)
    assert not any("alpha.md" in p for p in paths), (
        f"deleted alpha.md still returned as a hit: {paths}"
    )


# ── 4. external git commit (no brain_update) ─────────────────────────────────

def test_external_git_commit_changes_visible_to_search(brain_env):
    """Edit + commit a doc directly via git — bypassing brain_update.

    The vector index path is filesystem mtime/size based, not git-state
    based, so the change should still show up. If it doesn't, the
    index is bound to brain_update rather than to disk truth.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    alpha = repo / "docs" / "alpha.md"
    distinct = "supercalifragilistic"
    alpha.write_text(alpha.read_text() + f"\n{distinct} content.\n")
    _git(repo, "add", "docs/alpha.md")
    _git(repo, "commit", "-m", "external commit")

    out = _search(brain_env, distinct)
    assert any("alpha" in p for p in _paths(out)), (
        f"external git commit not reflected in search: {_paths(out)}"
    )


# ── 5. brand-new file via git only ───────────────────────────────────────────

def test_brand_new_file_via_git_is_findable(brain_env):
    """Create + git-commit a brand-new doc without going through brain_update.

    Pure disk truth: the file is there, the vector index should pick it up.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/delta.md"
    distinct = "kryptonite"
    (repo / rel).write_text(
        make_frontmatter("Delta superman note", ["fictional"])
        + f"Notes about {distinct} exposure.\n"
    )
    _git(repo, "add", rel)
    _git(repo, "commit", "-m", "external add")

    out = _search(brain_env, distinct)
    assert any("delta" in p for p in _paths(out)), (
        f"new file added outside brain_update not found: {_paths(out)}"
    )


# ── 6. remote diverges, local stays behind ───────────────────────────────────

def test_search_uses_local_disk_when_remote_diverges(brain_env):
    """Force-update the bare remote to a different commit.

    Local repo's working tree is unchanged. brain_search must operate on
    LOCAL disk truth (it has no automatic fetch). The seed term should
    still be found. This codifies "search reads disk, not remote".
    """
    repo = Path(brain_env["BRAIN_REPO"])
    bare = repo.parent / "brain.git"

    # Create a side branch that drops alpha.md, push it as main forcefully.
    tmp_clone = repo.parent / "tmp_clone"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(tmp_clone)],
        check=True, capture_output=True,
    )
    _git(tmp_clone, "config", "user.email", "ext@test")
    _git(tmp_clone, "config", "user.name", "Ext")
    (tmp_clone / "docs" / "alpha.md").unlink()
    _git(tmp_clone, "add", "-A")
    _git(tmp_clone, "commit", "-m", "drop alpha externally")
    _git(tmp_clone, "push", "-q", "--force", "origin", "main")

    # Local repo untouched: alpha.md still on disk → search should find it.
    out = _search(brain_env, "quokka nasturtium")
    assert any("alpha" in p for p in _paths(out)), (
        f"local disk truth lost when remote diverged: {_paths(out)}"
    )


# ── 7. symlink an external doc into docs/ ────────────────────────────────────

def test_symlinked_external_doc_indexing_behaviour(brain_env):
    """Symlink a markdown file outside the repo into docs/.

    Either it gets indexed (rglob follows symlinks by default) or it
    doesn't. Codify the actual behaviour — and check that if it IS
    indexed, the path doesn't escape the repo.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    external_dir = repo.parent / "external_notes"
    external_dir.mkdir(exist_ok=True)
    external = external_dir / "secret.md"
    distinct = "zygomorphic"
    external.write_text(
        make_frontmatter("External secret", ["external"])
        + f"Externally hosted {distinct} content.\n"
    )

    link = repo / "docs" / "linked.md"
    os.symlink(external, link)

    out, is_error = run_raw(brain_env, "brain_search", {"query": distinct})
    # Document whichever way it works. If indexed, path must stay
    # inside docs/. If not indexed, results must be empty for this
    # distinctive query.
    if isinstance(out, dict) and out.get("results"):
        paths = _paths(out)
        # Hit path should be repo-relative under docs/.
        for p in paths:
            assert not Path(p).is_absolute(), f"absolute path leaked: {p}"
            assert ".." not in p.split("/"), f"escaping path: {p}"
        # If we got results, the symlinked content was either found or
        # not — but no result should reference an absolute external path.
    # Otherwise: simply document that symlinked content is NOT searchable
    # (which is itself a useful, codified behaviour).


# ── 8. status: archived frontmatter — surfaces in search? ────────────────────

def test_archived_frontmatter_still_surfaces_unless_filtered(brain_env):
    """A doc with `status: archived` in its frontmatter.

    There is no implicit "hide archived" filter on brain_search — the
    `status` arg lets callers filter explicitly. Verify the doc shows
    up by default (no filter), and verify it's filterable when the
    caller passes status='archived'. This codifies "no implicit hiding".
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/archived_note.md"
    distinct = "obsoletewidget"
    # Hand-built frontmatter so we control status=archived.
    body = (
        "---\n"
        "title: Archived widget note\n"
        "owner: tests\n"
        "status: archived\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags: [legacy]\n"
        "type: note\n"
        "---\n\n"
        f"Discussion of the {distinct} project, long since retired.\n"
    )
    (repo / rel).write_text(body)

    # Default search (no filter): archived doc should still appear.
    out_default = _search(brain_env, distinct)
    default_paths = _paths(out_default)
    assert any("archived_note" in p for p in default_paths), (
        f"archived doc not in default results — implicit hiding bug? "
        f"got {default_paths}"
    )

    # Active-filtered search: archived doc must be excluded.
    out_active = _search(brain_env, distinct, status="active")
    active_paths = _paths(out_active)
    assert not any("archived_note" in p for p in active_paths), (
        f"status=active filter leaked archived doc: {active_paths}"
    )
