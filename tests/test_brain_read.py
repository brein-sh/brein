"""brain_read MCP tool: full-doc retrieval, telemetry, path safety."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import brain_env, call_tool, make_frontmatter, run, run_raw  # noqa: F401


def _seed_long_doc(env, rel: str, body_len: int = 5000) -> str:
    """Write a doc larger than the old 2500-char evidence cap, commit, reindex."""
    repo = Path(env["BRAIN_REPO"])
    body = make_frontmatter("Long doc", ["test"]) + ("x " * (body_len // 2)) + "\n"
    (repo / rel).write_text(body)
    subprocess.run(["git", "-C", str(repo), "add", rel], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", f"add {rel}"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "brain_mcp.cli", "index", "build"],
        env=env, check=True, capture_output=True,
    )
    return body


def test_brain_read_returns_full_doc(brain_env):
    """brain_read returns the whole file body — no 2500-char silent truncation."""
    body = _seed_long_doc(brain_env, "docs/longread.md", body_len=6000)
    out = run(brain_env, "brain_read", {"file_path": "docs/longread.md"})
    assert out["path"] == "docs/longread.md"
    assert out["content"] == body, "brain_read truncated or altered file body"
    assert out["truncated"] is False
    # Frontmatter parsed out for the caller.
    assert out["frontmatter"].get("title") == "Long doc"


def test_brain_read_missing_path_errors_cleanly(brain_env):
    out, _ = run_raw(brain_env, "brain_read", {"file_path": "docs/does-not-exist.md"})
    assert isinstance(out, dict) and "error" in out
    assert "not found" in out["error"]


def test_brain_read_rejects_path_traversal(brain_env):
    """Absolute paths + escapes outside the repo must be refused by _safe_path."""
    out, is_err = run_raw(brain_env, "brain_read", {"file_path": "../../etc/passwd"})
    # _safe_path raises ValueError → server surfaces as MCP error.
    assert is_err or (isinstance(out, dict) and "error" in out), out

    out, is_err = run_raw(brain_env, "brain_read", {"file_path": "/etc/passwd"})
    assert is_err or (isinstance(out, dict) and "error" in out), out


def test_brain_read_logs_telemetry(brain_env):
    """brain_read appends a 'read' row to the retrieval log."""
    _seed_long_doc(brain_env, "docs/teleread.md", body_len=300)
    log_path = Path(brain_env["BRAIN_RETRIEVAL_LOG"])
    before = log_path.read_text().count("\n") if log_path.exists() else 0
    run(brain_env, "brain_read", {"file_path": "docs/teleread.md"})
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    read_rows = [r for r in rows if r.get("kind") == "read"]
    assert read_rows, "no 'read' telemetry row produced"
    last = read_rows[-1]
    assert "docs/teleread.md" in last.get("hits", [])
    assert "docs/teleread.md" in last.get("used_docs", [])


def test_brain_read_max_chars_arg_caps_body(brain_env):
    """Caller can shrink the body with max_chars; total_chars reports the full size."""
    body = _seed_long_doc(brain_env, "docs/cap.md", body_len=4000)
    out = run(brain_env, "brain_read", {"file_path": "docs/cap.md", "max_chars": 500})
    assert len(out["content"]) == 500
    assert out["truncated"] is True
    assert out["total_chars"] == len(body)


def test_brain_read_max_chars_zero_means_uncapped(brain_env):
    """max_chars=0 disables the cap, even when body exceeds the default."""
    body = _seed_long_doc(brain_env, "docs/big.md", body_len=4000)
    out = run(brain_env, "brain_read", {"file_path": "docs/big.md", "max_chars": 0})
    assert out["content"] == body
    assert out["truncated"] is False
