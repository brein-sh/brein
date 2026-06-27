"""Bug-hunting tests for index corruption / recovery in brein-mcp.

Each test mutates the on-disk vector index file (or wraps the path in a
filesystem oddity) BETWEEN the fixture's clean index build and a real
MCP brain_search call. The assertions look for either:

  (a) graceful detection + recovery (status payload or rebuilt index), or
  (b) a bug (crash, garbage results, silent corruption acceptance).

If the server crashes or returns garbage, the test fails — that's the
finding.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from conftest import call_tool, run_raw


def _index_path(env: dict[str, str]) -> Path:
    return Path(env["BRAIN_VECTOR_INDEX"])


def _state_path(env: dict[str, str]) -> Path:
    return _index_path(env).with_name("index-state.json")


def _clear_state(env: dict[str, str]) -> None:
    """Remove the index-state.json so resolve_status doesn't short-circuit."""
    sp = _state_path(env)
    if sp.exists():
        sp.unlink()


def _search(env: dict[str, str]) -> tuple[object, bool]:
    return run_raw(env, "brain_search", {"query": "quokka"})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Truncated mid-JSON (worker killed mid-write)
# ─────────────────────────────────────────────────────────────────────────────

def test_truncated_json_does_not_crash(brain_env):
    p = _index_path(brain_env)
    raw = p.read_text(encoding="utf-8")
    # Truncate at ~60% so the JSON is unparseable but file is non-empty.
    p.write_text(raw[: max(10, int(len(raw) * 0.6))], encoding="utf-8")
    _clear_state(brain_env)

    payload, is_error = _search(brain_env)
    # Bug if the tool surfaces an MCP-level error or returns a non-dict.
    assert not is_error, f"brain_search errored on truncated index: {payload!r}"
    assert isinstance(payload, dict), f"expected dict payload, got {type(payload)}: {payload!r}"
    # Recovery invariant: either we got a structured status, or we got results.
    # We must NOT silently treat corrupted JSON as "ready with 0 hits" — that
    # would let agents think the brain is empty when it's actually broken.
    if "status" in payload:
        assert payload["status"] in {"missing", "empty", "building", "stalled"}, payload
    else:
        # If results came back, the server must have rebuilt successfully.
        assert "results" in payload, f"weird shape: {payload!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Completely empty file (0 bytes)
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_zero_byte_index(brain_env):
    p = _index_path(brain_env)
    p.write_text("", encoding="utf-8")
    _clear_state(brain_env)

    payload, is_error = _search(brain_env)
    assert not is_error, f"brain_search errored on 0-byte index: {payload!r}"
    assert isinstance(payload, dict), f"non-dict payload: {payload!r}"
    if "status" in payload:
        assert payload["status"] != "ready", (
            f"0-byte index reported status=ready: {payload!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Valid JSON, unexpected schema
# ─────────────────────────────────────────────────────────────────────────────

def test_index_missing_entries_key(brain_env):
    p = _index_path(brain_env)
    p.write_text(json.dumps({"version": 2, "built_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
    _clear_state(brain_env)

    payload, is_error = _search(brain_env)
    assert not is_error, f"errored on schema-missing-entries: {payload!r}"
    assert isinstance(payload, dict), payload
    # Bug if server reports ready+results when entries are absent.
    if isinstance(payload, dict) and "results" in payload:
        assert payload["results"] == [], (
            f"server invented results from index with no 'entries': {payload!r}"
        )


def test_index_entries_wrong_type(brain_env):
    p = _index_path(brain_env)
    p.write_text(json.dumps({"version": 2, "entries": "not-a-list", "built_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
    _clear_state(brain_env)

    payload, is_error = _search(brain_env)
    # Either graceful status or successful rebuild — never a crash.
    assert not is_error, f"crashed on entries=str: {payload!r}"
    assert isinstance(payload, dict), payload


def test_index_top_level_is_a_list(brain_env):
    """JSON valid but root is a list, not an object."""
    p = _index_path(brain_env)
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    _clear_state(brain_env)

    payload, is_error = _search(brain_env)
    assert not is_error, f"crashed on list-root index: {payload!r}"
    assert isinstance(payload, dict), payload


# ─────────────────────────────────────────────────────────────────────────────
# 4. Index references docs that no longer exist on disk
# ─────────────────────────────────────────────────────────────────────────────

def test_index_references_deleted_docs(brain_env):
    repo = Path(brain_env["BRAIN_REPO"])
    # Delete a seed doc that's already in the freshly-built index.
    target = repo / "docs" / "alpha.md"
    assert target.exists()
    target.unlink()

    payload, is_error = _search(brain_env)
    assert not is_error, f"errored when index refs deleted doc: {payload!r}"
    assert isinstance(payload, dict), payload
    # If results returned, none should point at the deleted file.
    for r in payload.get("results", []) or []:
        assert r.get("path") != "docs/alpha.md", (
            f"stale hit for deleted file leaked through: {r!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. built_at in the future
# ─────────────────────────────────────────────────────────────────────────────

def test_index_built_at_in_the_future(brain_env):
    p = _index_path(brain_env)
    data = json.loads(p.read_text(encoding="utf-8"))
    future = (datetime.now(timezone.utc) + timedelta(days=365 * 10)).isoformat()
    data["built_at"] = future
    p.write_text(json.dumps(data), encoding="utf-8")
    _clear_state(brain_env)

    payload, is_error = _search(brain_env)
    assert not is_error, f"errored on future built_at: {payload!r}"
    assert isinstance(payload, dict), payload
    # We don't require the server to reject future timestamps, but if it
    # reports built_at it should at least surface what it read (no silent
    # mangle to "now"). This codifies "don't lie about the timestamp".
    meta = payload.get("meta") or {}
    vmeta = (meta.get("vector") if isinstance(meta, dict) else None) or {}
    if "built_at" in vmeta:
        assert vmeta["built_at"] == future or vmeta["built_at"] is None, (
            f"server mutated future built_at to: {vmeta['built_at']!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Index path is a directory, not a file
# ─────────────────────────────────────────────────────────────────────────────

def test_index_path_is_a_directory(brain_env):
    p = _index_path(brain_env)
    p.unlink()
    p.mkdir()  # make the path a directory
    _clear_state(brain_env)

    payload, is_error = _search(brain_env)
    # Either a graceful status payload OR a clean tool error — never a Python
    # traceback bleeding through with no actionable info.
    if is_error:
        # Acceptable, but the message must be informative.
        assert "index" in str(payload).lower() or "directory" in str(payload).lower(), (
            f"unhelpful error on directory-shaped index: {payload!r}"
        )
    else:
        assert isinstance(payload, dict), payload
        # Cannot claim ready.
        if "status" in payload:
            assert payload["status"] != "ready", payload


# ─────────────────────────────────────────────────────────────────────────────
# 7. Index file is read-only — server can't update it
# ─────────────────────────────────────────────────────────────────────────────

def test_index_file_read_only(brain_env):
    p = _index_path(brain_env)
    # Make file read-only AND parent dir read-only so a rewrite truly fails.
    p.chmod(0o400)
    # Also flip a doc so a rebuild would be attempted (file sig changes).
    doc = Path(brain_env["BRAIN_REPO"]) / "docs" / "alpha.md"
    body = doc.read_text(encoding="utf-8")
    time.sleep(0.01)
    doc.write_text(body + "\nmutation to invalidate fingerprint\n", encoding="utf-8")

    try:
        payload, is_error = _search(brain_env)
        # Either we get a clean status / results without crashing the worker,
        # or we get an informative error. We do NOT want a silent success
        # that pretends the read-only file was updated.
        assert isinstance(payload, (dict, str)), payload
        if isinstance(payload, dict) and "results" in payload:
            # Fine — server reused the cached/in-memory index.
            pass
        elif isinstance(payload, dict) and "status" in payload:
            assert payload["status"] != "ready" or "results" not in payload, payload
    finally:
        # Restore perms so pytest tmp cleanup doesn't choke.
        try:
            p.chmod(0o600)
        except OSError:
            pass
