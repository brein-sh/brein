"""Self-improvement loop: trigger arithmetic, loss filtering, agent integration."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys

import pytest


def _write_eval_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_count_ab_runs_real_schema(monkeypatch, tmp_path):
    """Real eval-log.jsonl rows: A/B verdicts have NO `kind` field, only a
    top-level `verdict`. Only gate_skipped rows have `kind`."""
    from brain_mcp import evolve
    log = tmp_path / "eval-log.jsonl"
    _write_eval_log(log, [
        {"verdict": "brain_better"},
        {"kind": "gate_skipped", "verdict": None},
        {"verdict": "tie"},
        {"verdict": "no_brain_better"},
        {"kind": "gate_skipped", "verdict": None},
        {"verdict": "brain_better"},
    ])
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", log)
    assert evolve._count_ab_runs() == 4


def test_read_recent_losses_filters_to_no_brain_wins(monkeypatch, tmp_path):
    from brain_mcp import evolve
    log = tmp_path / "eval-log.jsonl"
    _write_eval_log(log, [
        {"verdict": "brain_better", "question": "q1"},
        {"verdict": "no_brain_better", "question": "q2"},
        {"verdict": "tie", "question": "q3"},
        {"verdict": "no_brain_better", "question": "q4"},
    ])
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", log)
    losses = evolve._read_recent_losses(limit=50)
    assert [l["question"] for l in losses] == ["q2", "q4"]


def test_read_recent_losses_respects_limit(monkeypatch, tmp_path):
    from brain_mcp import evolve
    log = tmp_path / "eval-log.jsonl"
    _write_eval_log(log, [
        {"verdict": "no_brain_better", "question": f"q{i}"}
        for i in range(10)
    ])
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", log)
    losses = evolve._read_recent_losses(limit=3)
    assert [l["question"] for l in losses] == ["q7", "q8", "q9"]


def test_maybe_trigger_fires_on_multiple_of_50(monkeypatch):
    from brain_mcp import evolve
    monkeypatch.delenv(evolve.EVOLVE_GUARD_ENV, raising=False)
    monkeypatch.setattr(evolve, "_count_ab_runs", lambda: 50)
    spawned: list = []
    monkeypatch.setattr(evolve, "_spawn_detached", lambda: spawned.append(1) or 12345)
    pid = evolve.maybe_trigger_after_ab()
    assert pid == 12345 and len(spawned) == 1


def test_maybe_trigger_silent_on_non_multiple(monkeypatch):
    from brain_mcp import evolve
    monkeypatch.delenv(evolve.EVOLVE_GUARD_ENV, raising=False)
    monkeypatch.setattr(evolve, "_count_ab_runs", lambda: 49)
    monkeypatch.setattr(
        evolve, "_spawn_detached",
        lambda: pytest.fail("must not spawn when count is not a multiple"),
    )
    assert evolve.maybe_trigger_after_ab() is None


def test_maybe_trigger_silent_on_zero(monkeypatch):
    """0 % 50 == 0 must NOT fire — there's nothing to improve from."""
    from brain_mcp import evolve
    monkeypatch.delenv(evolve.EVOLVE_GUARD_ENV, raising=False)
    monkeypatch.setattr(evolve, "_count_ab_runs", lambda: 0)
    monkeypatch.setattr(
        evolve, "_spawn_detached",
        lambda: pytest.fail("must not spawn when count is 0"),
    )
    assert evolve.maybe_trigger_after_ab() is None


def test_maybe_trigger_respects_guard_env(monkeypatch):
    """An evolve worker that itself calls into the brain MUST NOT recurse."""
    from brain_mcp import evolve
    monkeypatch.setenv(evolve.EVOLVE_GUARD_ENV, "1")
    monkeypatch.setattr(evolve, "_count_ab_runs", lambda: 50)
    monkeypatch.setattr(
        evolve, "_spawn_detached",
        lambda: pytest.fail("must not spawn when guard env is set"),
    )
    assert evolve.maybe_trigger_after_ab() is None


def test_spawn_detached_uses_module_invocation(monkeypatch, tmp_path):
    """Same launchd-PATH lesson as consistency: invoke via sys.executable -m."""
    from brain_mcp import evolve
    monkeypatch.setattr(evolve, "EVOLVE_LOG_PATH", tmp_path / "evolve.jsonl")
    captured: dict = {}

    class FakeProc:
        pid = 42
    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env", {})
        return FakeProc()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    evolve._spawn_detached()
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1:4] == ["-m", "brain_mcp.cli", "evolve"]
    assert captured["cmd"][4] == "run"
    # Recursion guard goes through the env so the spawned worker won't
    # re-fire evolve when its own eval write completes.
    assert captured["env"].get(evolve.EVOLVE_GUARD_ENV) == "1"


def test_run_evolve_appends_result_row(monkeypatch, tmp_path):
    """End-to-end with a stubbed agent: read losses → ask_llm → log row."""
    from brain_mcp import evolve, shared

    eval_log = tmp_path / "eval-log.jsonl"
    _write_eval_log(eval_log, [
        {"verdict": "no_brain_better",
         "question": "where does the SOR live?",
         "brain_answer": "abstract narrative",
         "no_brain_answer": "services/sor/router.py:L20-L80",
         "reason": "B had concrete paths"},
    ])
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", eval_log)
    monkeypatch.setattr(evolve, "EVOLVE_LOG_PATH", tmp_path / "evolve-log.jsonl")

    # Stub the agent.
    valid = json.dumps({
        "kind": "improved",
        "confidence": "high",
        "canonical_path": "docs/decisions/sor.md",
        "verified_refs_added": ["services/sor/router.py:L20-L80"],
        "edits_applied": True,
        "summary": "added 1 verified ref",
        "escalation_reason": None,
    })
    monkeypatch.setattr(shared, "ask_llm", lambda *a, **kw: (valid, "cli:fake", {}))
    monkeypatch.setattr(evolve, "_run_rechecks", lambda *a, **kw: {"fired": 0, "errored": 0, "total": 0, "detail": []})

    # Stub commit + push (test should not touch git).
    @contextlib.contextmanager
    def fake_lock():
        yield
    monkeypatch.setattr(shared, "_interprocess_write_lock", fake_lock)

    class R:
        def __init__(self, out="", rc=0):
            self.stdout = out
            self.returncode = rc

    seq = iter([
        R(" M docs/decisions/sor.md\n"),  # status: dirty
        R(""),                              # add -A
        R(""),                              # commit
        R(""),                              # push
        R("abc1234\n"),                     # rev-parse
    ])
    monkeypatch.setattr(shared, "_run_git", lambda args, **kw: next(seq))

    result = evolve.run_evolve(limit=50)
    assert result.losses_examined == 1
    assert result.losses_improved == 1
    assert result.losses_skipped == 0
    assert result.commit_sha == "abc1234"

    rows = evolve.read_log(limit=10)
    assert len(rows) == 1
    assert rows[0]["losses_improved"] == 1


def test_loss_end_row_includes_agent_summary(monkeypatch, tmp_path):
    """v0.5.28: agent's `summary` and `escalation_reason` must land in the
    loss_end progress row so reasons are visible live, not only at cycle_end."""
    from brain_mcp import evolve, shared

    eval_log = tmp_path / "eval-log.jsonl"
    _write_eval_log(eval_log, [
        {"verdict": "no_brain_better", "question": "q",
         "brain_answer": "x", "no_brain_answer": "y", "reason": "r"},
    ])
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", eval_log)
    monkeypatch.setattr(evolve, "EVOLVE_LOG_PATH", tmp_path / "evolve-log.jsonl")
    monkeypatch.setattr(evolve, "EVOLVE_PROGRESS_PATH", progress)

    skipped = json.dumps({
        "kind": "skipped", "confidence": "low",
        "canonical_path": None,
        "verified_refs_added": [],
        "edits_applied": False,
        "summary": "no canonical doc matches this question",
        "escalation_reason": None,
    })
    monkeypatch.setattr(shared, "ask_llm", lambda *a, **kw: (skipped, "cli:fake", {}))
    monkeypatch.setattr(evolve, "_run_rechecks", lambda *a, **kw: {"fired": 0, "errored": 0, "total": 0, "detail": []})

    evolve.run_evolve(limit=50)

    rows = [json.loads(l) for l in progress.read_text().splitlines() if l.strip()]
    end_rows = [r for r in rows if r["event"] == "loss_end"]
    assert end_rows[0]["summary"] == "no canonical doc matches this question"
    assert "escalation_reason" in end_rows[0]
    assert "canonical_path" in end_rows[0]


def test_run_evolve_parallelizes_across_losses(monkeypatch, tmp_path):
    """v0.5.28: 8 losses each sleeping 0.3s should finish in well under
    sequential time (2.4s). Parallelism cap is EVOLVE_PARALLELISM (default 8)."""
    import time as _t
    from brain_mcp import evolve, shared

    eval_log = tmp_path / "eval-log.jsonl"
    _write_eval_log(eval_log, [
        {"verdict": "no_brain_better", "question": f"q{i}",
         "brain_answer": "x", "no_brain_answer": "y", "reason": "r"}
        for i in range(8)
    ])
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", eval_log)
    monkeypatch.setattr(evolve, "EVOLVE_LOG_PATH", tmp_path / "evolve-log.jsonl")
    monkeypatch.setattr(evolve, "EVOLVE_PROGRESS_PATH", tmp_path / "progress.jsonl")
    monkeypatch.setattr(evolve, "EVOLVE_PARALLELISM", 8)

    skipped = json.dumps({
        "kind": "skipped", "confidence": "low",
        "canonical_path": None, "verified_refs_added": [],
        "edits_applied": False, "summary": "x", "escalation_reason": None,
    })
    def slow_ask(*a, **kw):
        _t.sleep(0.3)
        return (skipped, "cli:fake", {})
    monkeypatch.setattr(shared, "ask_llm", slow_ask)
    monkeypatch.setattr(evolve, "_run_rechecks", lambda *a, **kw: {"fired": 0, "errored": 0, "total": 0, "detail": []})

    t0 = _t.perf_counter()
    result = evolve.run_evolve(limit=50)
    wall = _t.perf_counter() - t0

    # Sequential would take 8 * 0.3 = 2.4s; parallel should be well under 1s.
    assert result.losses_examined == 8
    assert wall < 1.5, f"expected parallel < 1.5s, got {wall:.2f}s — likely serial regression"


def test_run_evolve_writes_per_loss_progress(monkeypatch, tmp_path):
    """Progress log gets cycle_start + per-loss start/end + cycle_end rows.
    Tailable via `tail -f ~/.brein/evolve-progress.jsonl`."""
    from brain_mcp import evolve, shared
    import contextlib

    eval_log = tmp_path / "eval-log.jsonl"
    _write_eval_log(eval_log, [
        {"verdict": "no_brain_better", "question": "q1",
         "brain_answer": "x", "no_brain_answer": "y", "reason": "r"},
        {"verdict": "no_brain_better", "question": "q2",
         "brain_answer": "x", "no_brain_answer": "y", "reason": "r"},
    ])
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", eval_log)
    monkeypatch.setattr(evolve, "EVOLVE_LOG_PATH", tmp_path / "evolve-log.jsonl")
    monkeypatch.setattr(evolve, "EVOLVE_PROGRESS_PATH", progress)

    skipped = json.dumps({
        "kind": "skipped", "confidence": "low",
        "canonical_path": None, "verified_refs_added": [],
        "edits_applied": False, "summary": "x", "escalation_reason": None,
    })
    monkeypatch.setattr(shared, "ask_llm", lambda *a, **kw: (skipped, "cli:fake", {}))
    monkeypatch.setattr(evolve, "_run_rechecks", lambda *a, **kw: {"fired": 0, "errored": 0, "total": 0, "detail": []})

    evolve.run_evolve(limit=50)

    rows = [json.loads(l) for l in progress.read_text().splitlines() if l.strip()]
    events = [r["event"] for r in rows]
    # cycle_start, then per-loss start/end pairs, then cycle_end.
    assert events[0] == "cycle_start"
    assert events[-1] == "cycle_end"
    assert events.count("loss_start") == 2
    assert events.count("loss_end") == 2
    # Cursor info on every loss_end row.
    end_rows = [r for r in rows if r["event"] == "loss_end"]
    assert [r["index"] for r in end_rows] == [1, 2]
    assert all("elapsed_s" in r and "running_totals" in r for r in end_rows)
    # All rows share one cycle_id.
    cycle_ids = {r["cycle_id"] for r in rows}
    assert len(cycle_ids) == 1


def test_cmd_evolve_does_not_nameerror_on_json(monkeypatch, capsys):
    """Regression: _cmd_evolve uses json.dumps; cli.py must import json.
    Caught in production v0.5.24 — `brein evolve run` died with
    NameError: name 'json' is not defined."""
    from brain_mcp import cli, evolve

    class FakeResult:
        def to_json(self):
            return {"evolve_id": "x", "losses_improved": 0}

    monkeypatch.setattr(evolve, "run_evolve", lambda limit=50: FakeResult())

    class Args:
        action = "run"
        limit = 50
        quiet = False

    rc = cli._cmd_evolve(Args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "evolve_id" in out  # confirms json.dumps actually ran


def test_run_evolve_commits_dirty_repo_even_when_zero_improved(monkeypatch, tmp_path):
    """v0.5.29 fix: a killed prior run leaves Edit-tool changes uncommitted.
    The next run will correctly skip ('docs already cover'), but it MUST
    still attempt the commit so the rescued edits land. Earlier evolve
    gated commit on improved>0 and silently lost the prior cycle's work."""
    from brain_mcp import evolve, shared
    import contextlib

    eval_log = tmp_path / "eval-log.jsonl"
    _write_eval_log(eval_log, [
        {"verdict": "no_brain_better", "question": "q",
         "brain_answer": "x", "no_brain_answer": "y", "reason": "r"},
    ])
    monkeypatch.setattr(evolve, "EVAL_LOG_PATH", eval_log)
    monkeypatch.setattr(evolve, "EVOLVE_LOG_PATH", tmp_path / "evolve-log.jsonl")
    monkeypatch.setattr(evolve, "EVOLVE_PROGRESS_PATH", tmp_path / "progress.jsonl")

    skipped = json.dumps({
        "kind": "skipped", "confidence": "low",
        "canonical_path": None, "verified_refs_added": [],
        "edits_applied": False, "summary": "x", "escalation_reason": None,
    })
    monkeypatch.setattr(shared, "ask_llm", lambda *a, **kw: (skipped, "cli:fake", {}))
    monkeypatch.setattr(evolve, "_run_rechecks", lambda *a, **kw: {"fired": 0, "errored": 0, "total": 0, "detail": []})
    # Skip the recheck pass entirely for this test (we exercise it elsewhere).
    monkeypatch.setattr(evolve, "_run_rechecks", lambda *a, **kw: {"fired": 0, "errored": 0, "total": 0, "detail": []})

    @contextlib.contextmanager
    def fake_lock():
        yield
    monkeypatch.setattr(shared, "_interprocess_write_lock", fake_lock)

    class R:
        def __init__(self, out="", rc=0):
            self.stdout = out
            self.returncode = rc
    # Repo is dirty (from a hypothetical killed prior run), so the new
    # cycle's commit path MUST fire and produce a SHA.
    seq = iter([
        R(" M docs/foo.md\n"),  # status: dirty
        R(""),                  # add -A
        R(""),                  # commit
        R(""),                  # push
        R("deadbeef\n"),        # rev-parse
    ])
    monkeypatch.setattr(shared, "_run_git", lambda args, **kw: next(seq))

    result = evolve.run_evolve(limit=50)
    assert result.losses_improved == 0
    assert result.commit_sha == "deadbeef", (
        "v0.5.28 would have left dirty repo uncommitted because improved=0; "
        "v0.5.29 commits any dirty state at end-of-cycle."
    )


def test_recheck_fires_run_ab_with_evolve_id_trigger(monkeypatch, tmp_path):
    """v0.5.29: after improvements, recheck the same losses so the eval log
    has paired before/after rows. Trigger MUST be 'evolve_recheck:<id>' so
    the UI can group by evolution."""
    from brain_mcp import evolve

    captured: list[dict] = []
    def fake_run_ab(question, evidence, trigger, qhash):
        captured.append({"question": question, "trigger": trigger, "qhash": qhash})
    import brain_mcp.eval as _eval_mod
    monkeypatch.setattr(_eval_mod, "_run_ab", fake_run_ab)
    monkeypatch.setattr(evolve, "_build_evidence_block", lambda q: "FAKE_EVIDENCE")
    monkeypatch.setattr(evolve, "EVOLVE_PROGRESS_PATH", tmp_path / "p.jsonl")

    losses = [
        {"question": "where does the SOR live?"},
        {"question": "what's the partner fee model?"},
    ]
    summary = evolve._run_rechecks(losses, evolve_id="evo12345", cycle_id="cyc")
    assert summary["fired"] == 2
    assert summary["errored"] == 0
    triggers = {c["trigger"] for c in captured}
    assert triggers == {"evolve_recheck:evo12345"}
    questions = {c["question"] for c in captured}
    assert questions == {l["question"] for l in losses}


def test_recheck_skips_empty_question(monkeypatch, tmp_path):
    from brain_mcp import evolve
    monkeypatch.setattr(evolve, "EVOLVE_PROGRESS_PATH", tmp_path / "p.jsonl")
    out = evolve._recheck_one({"question": ""}, evolve_id="x", cycle_id="c")
    assert out["fired"] is False
    assert out["reason"] == "empty_question"
