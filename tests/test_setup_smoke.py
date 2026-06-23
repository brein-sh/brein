"""Smoke checks: config round-trip + MCP snippet shape.

Run with: uv run pytest tests/test_setup_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

from brain_mcp import _user_config, mcp_snippet


def test_config_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_user_config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(_user_config, "CONFIG_PATH", tmp_path / "config.json")

    cfg = _user_config.BreinConfig(
        repo_path="/tmp/brain",
        retrieval_log="/tmp/log.jsonl",
        eval_enabled=True,
    )
    _user_config.save(cfg)
    loaded = _user_config.load()
    assert loaded.repo_path == "/tmp/brain"
    assert loaded.eval_enabled is True
    assert loaded.embedding_model == "BAAI/bge-small-en-v1.5"  # default preserved


def test_config_save_creates_backup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_user_config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(_user_config, "CONFIG_PATH", tmp_path / "config.json")

    _user_config.save(_user_config.BreinConfig(repo_path="/a"))
    _user_config.save(_user_config.BreinConfig(repo_path="/b"))
    assert (tmp_path / "config.json.bak").exists()
    assert json.loads((tmp_path / "config.json.bak").read_text())["repo_path"] == "/a"


def test_snippet_requires_repo_path() -> None:
    try:
        mcp_snippet.snippet(_user_config.BreinConfig(), "claude")
    except ValueError as e:
        assert "repo_path" in str(e)
    else:
        raise AssertionError("expected ValueError for empty repo_path")


def test_snippet_shape() -> None:
    cfg = _user_config.BreinConfig(repo_path="/tmp/brain", eval_enabled=True)
    out = json.loads(mcp_snippet.snippet(cfg, "claude"))
    server = out["mcpServers"]["brain"]
    assert "command" in server
    assert server["env"]["BRAIN_REPO"] == "/tmp/brain"
    assert server["env"]["BRAIN_EVAL_ENABLED"] == "1"


def test_snippet_unknown_client() -> None:
    cfg = _user_config.BreinConfig(repo_path="/tmp/brain")
    try:
        mcp_snippet.snippet(cfg, "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
