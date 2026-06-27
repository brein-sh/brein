"""Bug-hunting tests for path handling / symlinks / traversal in brain_update.

Every test here probes a specific way `_safe_path` / `_allowed_write_path`
(src/brain_mcp/shared.py) could let a write escape the repo, land in `.git`,
or bypass the allowed-prefix policy. NO happy paths.

A test FAILS when a guard is missing or weaker than expected. We do NOT fix
the underlying issue — failures are the finding.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import brain_env, call_tool, make_frontmatter, run_raw  # noqa: F401


def _body(title: str = "pwned doc") -> str:
    return make_frontmatter(title, ["sec-test"]) + "payload body\n"


def _repo(env) -> Path:
    return Path(env["BRAIN_REPO"])


def _is_error_payload(result) -> bool:
    """call_tool returns (text, is_error). brain_update also encodes errors
    as JSON `{"error": ...}` with is_error=False. Treat either as rejection."""
    parsed, is_error = result
    if is_error:
        return True
    if isinstance(parsed, dict) and "error" in parsed:
        return True
    return False


# ── Symlink attacks ─────────────────────────────────────────────────────────


def test_symlink_inside_docs_pointing_outside_repo(brain_env, tmp_path):
    """Pre-place `docs/escape.md` as a symlink to a file OUTSIDE the repo.
    Writing should be rejected; the outside file must remain untouched."""
    repo = _repo(brain_env)
    outside = tmp_path / "outside_target.md"
    outside.write_text("ORIGINAL\n")
    link = repo / "docs" / "escape.md"
    link.symlink_to(outside)

    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/escape.md",
        "content": _body(),
        "commit_message": "attack: symlink outside",
    })
    rejected = _is_error_payload(result)
    outside_content = outside.read_text()
    assert outside_content == "ORIGINAL\n", (
        f"FILE OUTSIDE REPO WAS OVERWRITTEN VIA SYMLINK. "
        f"content={outside_content!r}, result={result}"
    )
    assert rejected, f"symlink-out write should be rejected; got {result}"


def test_repo_root_docs_symlink_to_tmp(brain_env, tmp_path):
    """If an attacker (or stale state) makes `docs` itself a symlink to
    /tmp/elsewhere, do writes land in /tmp/elsewhere? We rename the existing
    docs dir, then create a symlink in its place."""
    repo = _repo(brain_env)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # Move the real docs dir aside so we can replace it with a symlink.
    real_docs = repo / "docs"
    moved = repo.parent / "docs_real"
    real_docs.rename(moved)
    (repo / "docs").symlink_to(elsewhere)

    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/landed.md",
        "content": _body(),
        "commit_message": "attack: docs symlink",
    })

    landed_outside = (elsewhere / "landed.md").exists()
    # Either the call rejects, or — if it succeeded — the file landed inside
    # `elsewhere` which is outside the repo. That's the bug.
    assert not landed_outside, (
        f"FILE WRITTEN OUTSIDE REPO via root `docs` symlink "
        f"-> {elsewhere/'landed.md'}. result={result}"
    )


def test_symlink_to_dotgit(brain_env):
    """Pre-place `docs/sneak` as a symlink to `.git`. Writing to
    `docs/sneak/config` should be rejected (it would corrupt git)."""
    repo = _repo(brain_env)
    (repo / "docs" / "sneak").symlink_to(repo / ".git")
    original_config = (repo / ".git" / "config").read_text()

    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/sneak/config",
        "content": _body(),
        "commit_message": "attack: dotgit via symlink",
    })

    after = (repo / ".git" / "config").read_text()
    assert after == original_config, (
        f".git/config was modified via symlink traversal! result={result}"
    )
    assert _is_error_payload(result), f"symlink-to-.git write should reject; got {result}"


# ── Traversal string variants ───────────────────────────────────────────────


def test_url_encoded_traversal(brain_env):
    """`docs/%2e%2e/passwd` — URL-encoded `..`. Path should be rejected
    OR treated as a literal weird filename, but never escape."""
    repo = _repo(brain_env)
    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/%2e%2e/passwd",
        "content": _body(),
        "commit_message": "attack: url-encoded",
    })
    # The file must not have landed in repo parent.
    assert not (repo.parent / "passwd").exists(), "URL-encoded traversal escaped!"


def test_backslash_traversal(brain_env):
    """`docs\\..\\..\\etc\\passwd` — on POSIX, backslash is a literal char,
    not a separator. Should be either rejected by allowed-prefix or land as
    a weird filename inside docs/, never escape."""
    repo = _repo(brain_env)
    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs\\..\\..\\etc\\passwd",
        "content": _body(),
        "commit_message": "attack: backslash",
    })
    # Must NOT have escaped.
    for victim in [repo.parent / "etc" / "passwd", Path("/etc/passwd_attack_marker")]:
        assert not victim.exists(), f"backslash traversal escaped to {victim}"


def test_absolute_path_rejected(brain_env):
    result = run_raw(brain_env, "brain_update", {
        "file_path": "/etc/passwd",
        "content": _body(),
        "commit_message": "attack: absolute",
    })
    assert _is_error_payload(result), f"absolute path should reject; got {result}"


def test_dot_path_rejected(brain_env):
    result = run_raw(brain_env, "brain_update", {
        "file_path": ".",
        "content": _body(),
        "commit_message": "attack: dot",
    })
    assert _is_error_payload(result), f"`.` should reject; got {result}"


def test_dotdot_path_rejected(brain_env):
    result = run_raw(brain_env, "brain_update", {
        "file_path": "..",
        "content": _body(),
        "commit_message": "attack: dotdot",
    })
    assert _is_error_payload(result), f"`..` should reject; got {result}"


def test_traversal_dotdot_segments(brain_env):
    """`docs/../../escape.md` should be rejected (escapes after resolve)."""
    repo = _repo(brain_env)
    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/../../escape.md",
        "content": _body(),
        "commit_message": "attack: ../..",
    })
    assert _is_error_payload(result), f"../.. should reject; got {result}"
    assert not (repo.parent.parent / "escape.md").exists()


# ── Exotic characters ───────────────────────────────────────────────────────


def test_embedded_null_byte(brain_env):
    """`docs/foo\\0.md` — null byte in path. Python's open() raises ValueError
    on embedded NULs; we want the MCP layer to reject cleanly, not crash."""
    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/foo\x00.md",
        "content": _body(),
        "commit_message": "attack: null byte",
    })
    # Either rejected gracefully or returns error payload. Must not raise
    # an uncaught exception that breaks the MCP session — call_tool returned,
    # so server didn't die. Just assert nothing weird got committed.
    parsed, is_error = result
    # A clean rejection is the desired behavior.
    assert _is_error_payload(result), f"null-byte path should reject cleanly; got {result}"


def test_non_ascii_path_cafe(brain_env):
    """`docs/café.md` — accented char. Should write+commit cleanly."""
    repo = _repo(brain_env)
    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/café.md",
        "content": _body("Cafe note"),
        "commit_message": "non-ascii path",
    })
    parsed, is_error = result
    assert not _is_error_payload(result), f"non-ascii should succeed; got {result}"
    assert (repo / "docs" / "café.md").exists()


def test_non_ascii_path_japanese(brain_env):
    """`docs/日本語.md`."""
    repo = _repo(brain_env)
    result = run_raw(brain_env, "brain_update", {
        "file_path": "docs/日本語.md",
        "content": _body("JP note"),
        "commit_message": "non-ascii jp",
    })
    assert not _is_error_payload(result), f"japanese path should succeed; got {result}"
    assert (repo / "docs" / "日本語.md").exists()


def test_long_path_component(brain_env):
    """>255 char filename. Most filesystems cap at 255 bytes per component
    (APFS does). Should be rejected cleanly, not crash."""
    long_name = "a" * 300 + ".md"
    result = run_raw(brain_env, "brain_update", {
        "file_path": f"docs/{long_name}",
        "content": _body(),
        "commit_message": "attack: long name",
    })
    # Either succeeds (fs allows it) or is rejected. Should not crash session.
    # If it succeeded silently with truncation, that's a finding.
    parsed, is_error = result
    # At minimum, the response should be parseable.
    assert parsed is not None


# ── Case collision (macOS APFS is case-insensitive by default) ──────────────


def test_case_collision_macos(brain_env):
    """Write `docs/CASEFOO.md`, then write `docs/casefoo.md`. On case-
    insensitive APFS the second write replaces the first silently, and
    git's index now has two entries pointing to the same physical file."""
    repo = _repo(brain_env)
    r1 = run_raw(brain_env, "brain_update", {
        "file_path": "docs/CASEFOO.md",
        "content": _body("UPPER"),
        "commit_message": "case upper",
    })
    assert not _is_error_payload(r1), f"first write failed: {r1}"

    r2 = run_raw(brain_env, "brain_update", {
        "file_path": "docs/casefoo.md",
        "content": _body("lower"),
        "commit_message": "case lower",
    })
    # Document what happened. On macOS case-insensitive fs, both names
    # resolve to one file but git may track them as two distinct entries.
    upper = repo / "docs" / "CASEFOO.md"
    lower = repo / "docs" / "casefoo.md"
    upper_text = upper.read_text() if upper.exists() else None
    lower_text = lower.read_text() if lower.exists() else None
    # Bug indicator: both git-tracked but identical content on disk.
    # We just report — don't assert collision is a hard failure; flag if
    # the second succeeded yet the first's content is gone.
    if not _is_error_payload(r2):
        # If second write succeeded, we expect the second content to win on
        # disk. The original UPPER content should be gone.
        if upper_text and "UPPER" in upper_text and lower_text and "lower" in lower_text:
            pytest.fail(
                "CASE COLLISION: case-insensitive fs but both files have "
                "distinct content — impossible state, indicates a bug or "
                "case-sensitive fs (then test is moot)."
            )
