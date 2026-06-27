"""Bug-hunting tests for brain_update(mode='append').

Targets the append branch at server.py:401:
    new = old.rstrip() + "\n" + content.rstrip() + "\n"

Sharp invariants under test:
- append to non-existent doc: should error (validator) or create cleanly,
  not write a file beginning with a stray "\n" and missing frontmatter.
- newline hygiene: appended content cannot collide with prior last line.
- frontmatter integrity: appended content lands AFTER the closing ``---``,
  never inside the frontmatter block.
- empty append must not corrupt the file.
- a ``---`` line inside appended content must not be re-parsed as
  frontmatter on subsequent read.
- validator runs against the FULL concatenated file, not the chunk.
- the new content is findable via brain_search after the write.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import brain_env, make_frontmatter, run, run_raw  # noqa: F401


def _read(repo: Path, rel: str) -> str:
    return (repo / rel).read_text(encoding="utf-8")


def _frontmatter_block(text: str) -> str:
    """Return text from start through the second '---' line, inclusive."""
    m = re.match(r"^---\n.*?\n---\n", text, re.DOTALL)
    assert m, f"no frontmatter block found in:\n{text[:200]}"
    return m.group(0)


# ── 1. Append to a non-existent doc ──────────────────────────────────────────

def test_append_to_missing_doc_does_not_produce_leading_newline(brain_env):
    """If the file doesn't exist, append uses old='' → result begins with '\\n'.
    That orphan newline before the frontmatter could break validators that
    require the file to start with '---'."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/append_missing.md"
    body = make_frontmatter("Append missing doc", ["test"]) + "first line\n"

    out, _ = run_raw(brain_env, "brain_update", {
        "file_path": rel,
        "content": body,
        "commit_message": "test: append to missing",
        "mode": "append",
    })

    if isinstance(out, dict) and "error" in out:
        # Acceptable: validator rejected because file begins with "\n---".
        # Then the file must NOT exist on disk.
        assert not (repo / rel).exists(), \
            "rolled-back append left a phantom file on disk"
        return

    text = _read(repo, rel)
    assert not text.startswith("\n"), \
        f"append-to-missing produced a leading newline: {text[:40]!r}"
    assert text.startswith("---\n"), \
        f"append-to-missing did not start with frontmatter: {text[:40]!r}"


# ── 2. Frontmatter integrity ─────────────────────────────────────────────────

def test_append_lands_outside_frontmatter_block(brain_env):
    """Appended content must not be injected inside the frontmatter."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/alpha.md"
    before = _read(repo, rel)
    fm_before = _frontmatter_block(before)

    out = run(brain_env, "brain_update", {
        "file_path": rel,
        "content": "APPENDED_SENTINEL_42 wallaby skirmish.\n",
        "commit_message": "test: append outside frontmatter",
        "mode": "append",
    })
    assert "error" not in out, out

    after = _read(repo, rel)
    fm_after = _frontmatter_block(after)
    assert fm_before == fm_after, \
        "frontmatter block changed after append (content leaked inside)"
    assert "APPENDED_SENTINEL_42" in after.split("---\n", 2)[2], \
        "sentinel did not land in the body region"


# ── 3. Newline hygiene: no run-together lines ───────────────────────────────

def test_append_does_not_join_last_line_with_appended_line(brain_env):
    """Append twice; the last line of the first append and the first line of
    the second must NOT end up concatenated on the same line."""
    rel = "docs/alpha.md"
    repo = Path(brain_env["BRAIN_REPO"])

    run(brain_env, "brain_update", {
        "file_path": rel,
        "content": "LINE_ONE_MARKER",  # NB: no trailing newline.
        "commit_message": "test: append no-trailing-nl",
        "mode": "append",
    })
    run(brain_env, "brain_update", {
        "file_path": rel,
        "content": "LINE_TWO_MARKER",  # NB: no leading newline.
        "commit_message": "test: append again",
        "mode": "append",
    })

    text = _read(repo, rel)
    assert "LINE_ONE_MARKERLINE_TWO_MARKER" not in text, \
        "successive appends ran lines together (newline hygiene broken)"
    # And each marker should occupy its own line.
    assert re.search(r"^LINE_ONE_MARKER$", text, re.MULTILINE), text[-200:]
    assert re.search(r"^LINE_TWO_MARKER$", text, re.MULTILINE), text[-200:]


# ── 4. Empty append ──────────────────────────────────────────────────────────

def test_append_empty_string_is_noop_or_safe(brain_env):
    """Appending '' must not corrupt the file. Either rejected, or the
    on-disk content equals old.rstrip()+'\\n' (trailing whitespace only)."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/alpha.md"
    before = _read(repo, rel)

    out, _ = run_raw(brain_env, "brain_update", {
        "file_path": rel,
        "content": "",
        "commit_message": "test: empty append",
        "mode": "append",
    })

    after = _read(repo, rel)
    # Compare ignoring trailing whitespace — the substantive content must
    # be identical. If the file body shrank or gained junk, that's a bug.
    assert before.rstrip() == after.rstrip(), \
        f"empty append mutated body: before={before[-60:]!r} after={after[-60:]!r}"
    # And the file must still parse as a valid doc (frontmatter intact).
    assert after.startswith("---\n"), "empty append corrupted frontmatter start"


# ── 5. Appending content containing a '---' line ────────────────────────────

def test_appended_triple_dash_does_not_create_second_frontmatter(brain_env):
    """If appended content includes a line '---', a naive frontmatter parser
    on re-read might think a second frontmatter block exists. Our invariant:
    only ONE frontmatter block at file start, and appended '---' lands in body."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/alpha.md"

    payload = "section break:\n---\nmore body POSTDIVIDER_TOKEN here\n"
    out = run(brain_env, "brain_update", {
        "file_path": rel,
        "content": payload,
        "commit_message": "test: append with triple dash",
        "mode": "append",
    })
    assert "error" not in out, out

    after = _read(repo, rel)
    # Count standalone '---' lines.
    dash_lines = re.findall(r"^---$", after, re.MULTILINE)
    # Frontmatter contributes exactly 2. Body now contributes at least 1.
    # Bug signal: if any tooling later treats the third '---' as closing a
    # new frontmatter block, that's silent corruption. We codify "only the
    # first two '---' lines bound the frontmatter".
    assert len(dash_lines) >= 3, \
        f"expected at least 3 '---' lines (2 frontmatter + 1 body), got {len(dash_lines)}"
    fm = _frontmatter_block(after)
    assert "POSTDIVIDER_TOKEN" not in fm, \
        "appended '---' caused content to be absorbed into frontmatter block"


# ── 6. Validator runs on the FULL concatenated file ─────────────────────────

def test_append_garbage_to_valid_doc_does_not_silently_succeed(brain_env):
    """The validator must check the full file post-append. Appending content
    that, on its own, looks like a valid doc fragment is fine — but appending
    something that breaks the doc must roll back. Here we append a duplicate
    frontmatter block; the resulting doc has two '---'-delimited blocks at
    the top which a strict validator should reject. If it doesn't reject AND
    we end up with a corrupted committed file — that's the bug."""
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/alpha.md"
    before = _read(repo, rel)

    second_fm = make_frontmatter("Second header", ["dup"]) + "second body.\n"
    out, _ = run_raw(brain_env, "brain_update", {
        "file_path": rel,
        "content": second_fm,
        "commit_message": "test: append duplicate frontmatter",
        "mode": "append",
    })

    after = _read(repo, rel)
    if isinstance(out, dict) and "error" in out:
        # Validator caught it — file must be unchanged.
        assert after == before, "rolled-back append still mutated the file"
        return

    # If the write was accepted, the resulting file must still parse cleanly:
    # exactly ONE frontmatter block at the start.
    fm_blocks = re.findall(r"(?m)^---\n", after)
    # If validator accepted a file with 4 '---' fences at the top (two FMs),
    # that is the bug we're surfacing.
    assert fm_blocks.count("---\n") <= 2 or after.count("---\n", 0, 600) <= 2, \
        f"validator accepted a doc with duplicate frontmatter:\n{after[:400]}"


# ── 7. Findability after append ──────────────────────────────────────────────

def test_appended_content_is_findable_by_search(brain_env):
    """A unique token added via append must be searchable. If the index
    isn't refreshed after an append, this fails."""
    rel = "docs/alpha.md"
    unique = "ZQXJBWUNIQUETOKEN_marmoset_serenade"
    out = run(brain_env, "brain_update", {
        "file_path": rel,
        "content": f"{unique} appears once in the corpus.\n",
        "commit_message": "test: append unique token",
        "mode": "append",
    })
    assert "error" not in out, out

    found = run(brain_env, "brain_search", {"query": unique})
    if found.get("status") != "ready":
        pytest.skip(f"index not ready: {found.get('status')}")
    paths = [r["path"] for r in found.get("results", [])]
    assert rel in paths, \
        f"appended unique token not retrievable via search: paths={paths}"
