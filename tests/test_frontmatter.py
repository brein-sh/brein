"""Bug-hunting E2E tests for frontmatter parsing/validation.

Each test attempts a brain_update with malformed/edge-case frontmatter and
asserts that the validator catches it (rolls back). Tests that fail indicate
the parser/validator silently accepted something it shouldn't have — those
are findings, not failures to fix.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conftest import brain_env, call_tool, make_frontmatter, run_raw  # noqa: F401


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _attempt_write(brain_env, rel: str, content: str):
    """Attempt brain_update; return (parsed_or_text, is_error)."""
    return run_raw(brain_env, "brain_update", {
        "file_path": rel,
        "content": content,
        "commit_message": f"test: frontmatter probe {rel}",
    })


def _assert_rejected(out, is_error, repo: Path, rel: str, head_before: str, label: str):
    """Assert the write was rejected (rolled back or surfaced as an error)."""
    rejected = (
        is_error
        or (isinstance(out, dict) and ("error" in out or out.get("rolled_back") is True))
    )
    assert rejected, f"{label}: silently accepted malformed frontmatter: {out!r}"
    assert not (repo / rel).exists(), f"{label}: file persisted on disk: {rel}"
    assert _head(repo) == head_before, f"{label}: HEAD advanced despite rejection"


# ── Structural malformations ─────────────────────────────────────────────────


def test_frontmatter_no_closing_fence(brain_env):
    """Opening `---` but no closing fence. Validator regex requires closing."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/no_close.md"
    head_before = _head(repo)
    content = (
        "---\n"
        "title: never closes\n"
        "owner: tests\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags: ['x']\n"
        "type: note\n"
        "body without closing fence\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    _assert_rejected(out, is_error, repo, rel, head_before, "no-closing-fence")


def test_frontmatter_empty_block(brain_env):
    """`---\\n---\\n` empty block — no fields at all."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/empty_fm.md"
    head_before = _head(repo)
    content = "---\n---\n\nbody.\n"
    out, is_error = _attempt_write(brain_env, rel, content)
    _assert_rejected(out, is_error, repo, rel, head_before, "empty-block")


def test_frontmatter_with_bom(brain_env):
    """U+FEFF BOM at start. validate_docs uses text.startswith('---'),
    which fails when BOM precedes the fence → 'missing frontmatter block'.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/bom_fm.md"
    head_before = _head(repo)
    content = "﻿" + make_frontmatter("BOM test", ["bom"]) + "body.\n"
    out, is_error = _attempt_write(brain_env, rel, content)
    _assert_rejected(out, is_error, repo, rel, head_before, "BOM-prefix")


def test_frontmatter_crlf_line_endings(brain_env):
    """Windows-style \\r\\n line endings. The closing-fence regex is
    `\\n---\\s*\\n` — should still match, but parser may keep \\r in values
    or fail other ways. Probe for round-trip integrity.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/crlf_fm.md"
    head_before = _head(repo)
    content = (make_frontmatter("CRLF doc", ["crlf"]) + "body.\n").replace("\n", "\r\n")
    out, is_error = _attempt_write(brain_env, rel, content)
    # If accepted, verify the on-disk file is sane; if rejected, that's also
    # a finding worth noting but consistent with strict parsing.
    if not (is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))):
        # Accepted — make sure the title was actually parsed (no \r residue).
        on_disk = (repo / rel).read_text()
        assert "CRLF doc" in on_disk
    else:
        _assert_rejected(out, is_error, repo, rel, head_before, "CRLF")


def test_frontmatter_tabs_for_indent(brain_env):
    """Tab-prefixed lines are skipped by the parser (line[0] in (' ', '\\t')),
    which means a tab-prefixed required field is treated as missing.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/tabs_fm.md"
    head_before = _head(repo)
    # Put 'type:' on a tab-indented line. Pattern-check uses `in text`, so it
    # passes the missing-field gate, but parser drops it → fm dict missing it.
    content = (
        "---\n"
        "title: tabby\n"
        "owner: tests\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags: ['t']\n"
        "\ttype: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    # Either rejected (good) or accepted (bug: indented field silently dropped).
    if not (is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))):
        pytest.fail(
            f"tab-indented field silently accepted; parser skips indented "
            f"lines so 'type' may be missing from parsed fm: {out!r}"
        )


# ── Value-shape malformations ────────────────────────────────────────────────


def test_frontmatter_value_contains_colon(brain_env):
    """`title: foo: bar` — does the value preserve the second colon?"""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/colon_value.md"
    head_before = _head(repo)
    content = (
        "---\n"
        "title: foo: bar baz\n"
        "owner: tests\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags: ['c']\n"
        "type: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    # This should be accepted (valid YAML-ish). Verify file landed.
    if is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back"))):
        pytest.fail(f"colon-in-value rejected unexpectedly: {out!r}")
    assert (repo / rel).exists(), "colon-in-value: file not written"


def test_frontmatter_multiline_yaml_tags(brain_env):
    """tags as a YAML block list:
        tags:
          - foo
          - bar
    Parser skips indented lines, so tags value parses as empty string.
    Required-pattern check uses `in text`, so it passes. Probe for write.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/multiline_tags.md"
    head_before = _head(repo)
    content = (
        "---\n"
        "title: multiline tags\n"
        "owner: tests\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags:\n"
        "  - foo\n"
        "  - bar\n"
        "type: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    # Should accept — but the parsed tags value is the empty string,
    # losing all tag info silently. That's a latent bug regardless of accept/reject.
    accepted = not (
        is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))
    )
    assert accepted, f"multiline tags rejected: {out!r}"
    # We can't introspect parsed fm from outside, but document the behavior:
    # if accepted, the file lands with tag info that the parser will lose.
    assert (repo / rel).exists()


# ── Taxonomy / value-domain checks ───────────────────────────────────────────


def test_review_cycle_uppercase_rejected(brain_env):
    """`review_cycle: ANNUAL` — REVIEW_CYCLE_DAYS keys are lowercase, and
    check_staleness lowercases the value before lookup. So uppercase
    should be accepted. Probe for the actual behavior.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/cycle_upper.md"
    head_before = _head(repo)
    content = (
        "---\n"
        "title: uppercase cycle\n"
        "owner: tests\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: ANNUAL\n"
        "tags: ['x']\n"
        "type: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    accepted = not (
        is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))
    )
    # Expected: accepted (lowercase normalization happens).
    assert accepted, f"uppercase review_cycle rejected — possible strictness bug: {out!r}"


def test_review_cycle_garbage_rejected(brain_env):
    """`review_cycle: hourly` is not in REVIEW_CYCLE_DAYS → should error."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/cycle_bad.md"
    head_before = _head(repo)
    content = (
        "---\n"
        "title: bad cycle\n"
        "owner: tests\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: hourly\n"
        "tags: ['x']\n"
        "type: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    _assert_rejected(out, is_error, repo, rel, head_before, "garbage-cycle")


def test_status_not_in_allowed_set(brain_env):
    """`status: pending` not in ALLOWED_DOC_STATUSES → must reject."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/bad_status.md"
    head_before = _head(repo)
    content = (
        "---\n"
        "title: bad status\n"
        "owner: tests\n"
        "status: pending\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags: ['x']\n"
        "type: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    _assert_rejected(out, is_error, repo, rel, head_before, "bad-status")


def test_status_uppercase_accepted(brain_env):
    """`status: ACTIVE` — check_status lowercases before set check, so it
    should be accepted. Probe for actual behavior.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/status_upper.md"
    content = (
        "---\n"
        "title: uppercase status\n"
        "owner: tests\n"
        "status: ACTIVE\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags: ['x']\n"
        "type: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    accepted = not (
        is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))
    )
    assert accepted, f"uppercase status rejected — possible strictness inconsistency: {out!r}"


# ── Duplicate / collision cases ──────────────────────────────────────────────


def test_duplicate_keys_last_wins(brain_env):
    """Two `status:` lines — first 'archived', second 'active'. Parser uses
    plain dict assignment, so last write wins. If the FIRST status were a
    forbidden value but the second valid, validator would silently accept
    the doc — a real classification bug.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/dup_status.md"
    head_before = _head(repo)
    # First status: 'bogusvalue' (would normally be rejected).
    # Second status: 'active' (valid).
    content = (
        "---\n"
        "title: dup status\n"
        "owner: tests\n"
        "status: bogusvalue\n"
        "status: active\n"
        "last_reviewed: 2026-01-01\n"
        "review_cycle: annual\n"
        "tags: ['x']\n"
        "type: note\n"
        "---\n\nbody.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    # Bug-hunt expectation: if accepted, the first (invalid) status was
    # silently overridden by the second. Document by failing only when
    # accepted — meaning duplicate-status attack lets bad values through.
    accepted = not (
        is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))
    )
    if accepted:
        pytest.fail(
            "duplicate status keys: invalid first value silently overridden "
            "by valid second value. Validator does not catch duplicate keys."
        )


# ── Body-content edge cases ──────────────────────────────────────────────────


def test_stray_triple_dash_in_body(brain_env):
    """A line `---` later in the body shouldn't confuse the parser, but
    if the parser's end-finder is naive, it may truncate frontmatter.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/stray_dash.md"
    content = (
        make_frontmatter("stray dash", ["x"])
        + "Intro paragraph.\n\n"
        + "---\n\n"
        + "Section after a stray triple-dash line.\n"
    )
    out, is_error = _attempt_write(brain_env, rel, content)
    accepted = not (
        is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))
    )
    assert accepted, f"stray --- in body wrongly rejected: {out!r}"
    assert (repo / rel).exists()


def test_frontmatter_only_no_body(brain_env):
    """Frontmatter block followed by nothing. Should be accepted (or at
    least handled cleanly — not corrupt the repo).
    """
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/no_body.md"
    head_before = _head(repo)
    content = make_frontmatter("just frontmatter", ["x"])  # ends with \n\n
    out, is_error = _attempt_write(brain_env, rel, content)
    # Either outcome is acceptable as long as HEAD is consistent with outcome.
    accepted = not (
        is_error or (isinstance(out, dict) and ("error" in out or out.get("rolled_back")))
    )
    if accepted:
        assert (repo / rel).exists()
    else:
        assert not (repo / rel).exists()
        assert _head(repo) == head_before
