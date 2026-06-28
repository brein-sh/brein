"""Regression tests for the four silent-failure classes that bit us on
2026-06-28 (v0.5.16 → v0.5.20). Each test simulates the launchd-daemon
environment — restricted PATH, sys.argv[0] pointing at the server
launcher — that production hits but pytest's dev env hides.

Pattern: directly call the helper, patch what we don't want to actually
fork, assert the call shape. No real subprocesses, no real LLM calls.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest


# ── 1. spawn_detached must invoke `python -m brain_mcp.cli`, not a CLI
#       on PATH. (v0.5.18 fix — when daemon's PATH excludes ~/.local/bin,
#       the old code raised FileNotFoundError silently swallowed by server.py.)

def test_spawn_detached_uses_module_invocation_not_brein_on_path(monkeypatch, tmp_path):
    from brain_mcp import consistency

    monkeypatch.setattr(consistency, "QUEUE_PATH", tmp_path / "queue.jsonl")
    captured: dict = {}

    class FakeProc:
        pid = 99999

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    pid = consistency.spawn_detached("docs/test.md")
    assert pid == 99999

    cmd = captured["cmd"]
    # MUST use the running interpreter, not search PATH for `brein`.
    assert cmd[0] == sys.executable, f"expected sys.executable, got {cmd[0]!r}"
    assert cmd[1:4] == ["-m", "brain_mcp.cli", "consistency"], cmd
    assert cmd[-1] == "docs/test.md"
    # Detachment flags that let the worker survive the daemon exiting.
    assert captured["kwargs"]["start_new_session"] is True


def test_spawn_detached_works_when_brein_not_on_path(monkeypatch, tmp_path):
    """Daemon condition: PATH excludes everything but /usr/bin /bin."""
    from brain_mcp import consistency

    monkeypatch.setattr(consistency, "QUEUE_PATH", tmp_path / "queue.jsonl")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    # sys.argv[0] mimics the launchd-spawned daemon launcher.
    monkeypatch.setattr(sys, "argv", ["/some/path/brain-mcp"])

    captured: dict = {}
    class FakeProc:
        pid = 1
    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    consistency.spawn_detached("docs/x.md")
    # Pre-0.5.18 this would have been ["brein", ...] and Popen would
    # raise FileNotFoundError. Now it's sys.executable + module form.
    assert captured["cmd"][0] == sys.executable


# ── 2. _which_llm_cli must find claude/codex/gemini in known install
#       paths after shutil.which() misses. (v0.5.19 fix.)

def test_which_llm_cli_falls_back_to_known_paths_when_path_empty(
    monkeypatch, tmp_path,
):
    from brain_mcp import shared

    # Build a fake claude binary in a tmp location.
    fake_claude = tmp_path / "claude"
    fake_claude.write_text("#!/bin/sh\necho hi\n")
    fake_claude.chmod(0o755)

    # Stripped PATH, like launchd gives us.
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    # No CLI on the stripped PATH.
    monkeypatch.setattr(shared.shutil, "which", lambda name: None)
    # Probe list points at our fake.
    monkeypatch.setattr(shared, "_KNOWN_CLI_PATHS", {"claude": [str(fake_claude)]})
    # Preference order: claude first.
    monkeypatch.setattr(shared, "_llm_cli_preference", lambda: ["claude"])

    found = shared._which_llm_cli()
    assert found == str(fake_claude), (
        f"expected fallback to {fake_claude}, got {found!r}. "
        "Pre-0.5.19, this returned None inside the daemon."
    )


def test_which_llm_cli_returns_none_when_neither_path_nor_known_match(
    monkeypatch,
):
    from brain_mcp import shared
    monkeypatch.setattr(shared.shutil, "which", lambda name: None)
    monkeypatch.setattr(shared, "_KNOWN_CLI_PATHS", {})
    monkeypatch.setattr(shared, "_llm_cli_preference", lambda: ["claude"])
    assert shared._which_llm_cli() is None


def test_which_llm_cli_known_path_must_be_executable(monkeypatch, tmp_path):
    """Non-executable file at a known path should NOT be picked."""
    from brain_mcp import shared
    fake = tmp_path / "claude"
    fake.write_text("not executable")
    fake.chmod(0o644)
    monkeypatch.setattr(shared.shutil, "which", lambda name: None)
    monkeypatch.setattr(shared, "_KNOWN_CLI_PATHS", {"claude": [str(fake)]})
    monkeypatch.setattr(shared, "_llm_cli_preference", lambda: ["claude"])
    assert shared._which_llm_cli() is None


# ── 3. ask_llm with allowed_tools must pass --allowed-tools to claude
#       and not corrupt cwd/env. (Behavior contract for the agentic mode.)

def test_ask_llm_passes_allowed_tools_to_claude(monkeypatch):
    from brain_mcp import shared

    monkeypatch.setattr(shared, "_which_llm_cli", lambda: "/fake/claude")

    captured: dict = {}
    class R:
        returncode = 0
        stdout = json.dumps({"result": "ok", "usage": {}})
        stderr = ""
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        captured["cwd"] = kwargs.get("cwd")
        return R()
    monkeypatch.setattr(subprocess, "run", fake_run)

    text, backend, meta = shared.ask_llm(
        "test prompt",
        disable_brain=True,
        allowed_tools=["Read", "Grep", "Edit"],
        cwd="/repo",
    )

    assert "--allowed-tools" in captured["cmd"], captured["cmd"]
    idx = captured["cmd"].index("--allowed-tools")
    assert captured["cmd"][idx + 1] == "Read,Grep,Edit"
    assert captured["cwd"] == "/repo"
    assert captured["env"].get("BRAIN_EVAL_DISABLE") == "1"
    assert backend == "cli:/fake/claude+agentic"
    assert text == "ok"


def test_ask_llm_no_allowed_tools_means_no_agentic_flag(monkeypatch):
    from brain_mcp import shared
    monkeypatch.setattr(shared, "_which_llm_cli", lambda: "/fake/claude")

    captured: dict = {}
    class R:
        returncode = 0
        stdout = json.dumps({"result": "x", "usage": {}})
        stderr = ""
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: (captured.update(cmd=cmd) or R()),
    )

    shared.ask_llm("p", disable_brain=True)
    assert "--allowed-tools" not in captured["cmd"]


# ── 4. _judge_agentic must return None (→ stub finding) when the LLM
#       returns empty text. The pre-0.5.19 silent failure mode.

def test_judge_agentic_returns_none_on_empty_llm_output(monkeypatch):
    from brain_mcp import consistency, shared

    monkeypatch.setattr(
        shared, "ask_llm",
        lambda *a, **kw: ("", "stub-backend", {"error": "no_cli"}),
    )
    result = consistency._judge_agentic(
        "docs/new.md", "new doc content", neighbors=[{"path": "docs/old.md"}],
    )
    assert result is None


def test_judge_agentic_parses_valid_json(monkeypatch):
    from brain_mcp import consistency, shared

    valid = json.dumps({
        "kind": "supersede",
        "confidence": "high",
        "summary": "test",
        "canonical_path": "docs/canonical.md",
        "deprecated_paths": ["docs/loser.md"],
        "edits_applied": True,
        "escalation_reason": None,
    })
    monkeypatch.setattr(shared, "ask_llm", lambda *a, **kw: (valid, "cli:fake", {}))

    result = consistency._judge_agentic(
        "docs/new.md", "content", neighbors=[{"path": "docs/old.md"}],
    )
    assert result is not None
    assert result["kind"] == "supersede"
    assert result["canonical_path"] == "docs/canonical.md"


# ── 5. _commit_agent_edits must commit + push when files are dirty, and
#       no-op cleanly when they're not.

def test_commit_agent_edits_noop_when_clean(monkeypatch):
    from brain_mcp import consistency, shared
    import contextlib

    @contextlib.contextmanager
    def fake_lock():
        yield

    monkeypatch.setattr(shared, "_interprocess_write_lock", fake_lock)

    class R:
        def __init__(self, out=""):
            self.stdout = out
            self.returncode = 0
    calls: list = []
    def fake_run(args, **kw):
        calls.append(args)
        return R("")  # clean status, no diff
    monkeypatch.setattr(shared, "_run_git", fake_run)

    result = consistency._commit_agent_edits({"kind": "ok", "summary": "x"})
    assert result is None
    # Only the status check should have run.
    assert any("status" in a for a in calls), calls
    assert not any("commit" in a for a in calls), calls
    assert not any("push" in a for a in calls), calls


def test_commit_agent_edits_commits_and_pushes_when_dirty(monkeypatch):
    from brain_mcp import consistency, shared
    import contextlib

    @contextlib.contextmanager
    def fake_lock():
        yield
    monkeypatch.setattr(shared, "_interprocess_write_lock", fake_lock)

    class R:
        def __init__(self, out="", rc=0):
            self.stdout = out
            self.returncode = rc

    sequence = iter([
        R(" M docs/x.md\n"),   # status: dirty
        R(""),                  # add -A
        R(""),                  # commit
        R(""),                  # push
        R("deadbeef\n"),        # rev-parse
    ])
    calls: list = []
    def fake_run(args, **kw):
        calls.append(args)
        return next(sequence)
    monkeypatch.setattr(shared, "_run_git", fake_run)

    result = consistency._commit_agent_edits({
        "kind": "supersede",
        "summary": "x contradicts y",
        "canonical_path": "docs/y.md",
    })
    assert result is not None
    assert result["sha"] == "deadbeef"
    assert "consistency(supersede)" in result["message"]
    # Verify the actual sequence we expect.
    verbs = [a[0] if a else "" for a in calls]
    assert verbs == ["status", "add", "commit", "push", "rev-parse"], verbs


# ── 6. run_check returns None silently on 'ok' with no edits, but emits
#       a Finding when the agent escalates or auto-resolves.

def test_run_check_silent_on_ok(monkeypatch, tmp_path):
    from brain_mcp import consistency

    # Set up a fake brain repo with one doc.
    doc = tmp_path / "docs" / "x.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("---\ntitle: x\n---\nhello\n")

    monkeypatch.setattr(consistency, "REPO_PATH", tmp_path)
    monkeypatch.setattr(consistency, "QUEUE_PATH", tmp_path / "queue.jsonl")
    monkeypatch.setattr(
        consistency, "_find_neighbors",
        lambda *a, **kw: [{"path": "docs/y.md", "vector_score": 0.9, "vector_snippet": "..."}],
    )
    monkeypatch.setattr(
        consistency, "_judge_agentic",
        lambda *a, **kw: {
            "kind": "ok", "confidence": "high", "summary": "",
            "canonical_path": None, "deprecated_paths": [],
            "edits_applied": False, "escalation_reason": None,
        },
    )

    result = consistency.run_check("docs/x.md")
    assert result is None
    assert not (tmp_path / "queue.jsonl").exists()


def test_run_check_emits_finding_on_escalate(monkeypatch, tmp_path):
    from brain_mcp import consistency

    doc = tmp_path / "docs" / "x.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("---\ntitle: x\n---\nhello\n")

    monkeypatch.setattr(consistency, "REPO_PATH", tmp_path)
    monkeypatch.setattr(consistency, "QUEUE_PATH", tmp_path / "queue.jsonl")
    monkeypatch.setattr(
        consistency, "_find_neighbors",
        lambda *a, **kw: [{"path": "docs/y.md", "vector_score": 0.9}],
    )
    monkeypatch.setattr(
        consistency, "_judge_agentic",
        lambda *a, **kw: {
            "kind": "escalate", "confidence": "medium",
            "summary": "needs human",
            "canonical_path": None,
            "deprecated_paths": ["docs/y.md"],
            "edits_applied": False,
            "escalation_reason": "two source_of_truth claims",
        },
    )

    finding = consistency.run_check("docs/x.md")
    assert finding is not None
    assert finding.kind == "unresolved"
    assert finding.judge == "agentic"
    queue_lines = (tmp_path / "queue.jsonl").read_text().strip().splitlines()
    assert len(queue_lines) == 1
