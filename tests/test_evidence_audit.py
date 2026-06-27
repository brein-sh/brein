"""Bug-hunting tests for brain_evidence and brain_audit.

No happy-path coverage — every test asserts a sharp behaviour or hunts a gap.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import brain_env, run, run_raw, call_tool  # noqa: F401


# ─────────────────────────── brain_evidence ───────────────────────────


def _retrieval_log_lines(env) -> list[dict]:
    p = Path(env["BRAIN_RETRIEVAL_LOG"])
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_evidence_off_topic_score_is_meaningfully_lower(brain_env):
    """Off-topic question's top vector score must be meaningfully below on-topic.

    Vector search has no relevance threshold — it always returns the nearest
    neighbour. The real semantic invariant is *score degradation*: an off-topic
    query against the same corpus must produce a top score noticeably lower
    than a topically-aligned query. If this gap collapses, the embedder has
    regressed (or got replaced with something near-random), and grounded
    answers become indistinguishable from hallucinations.
    """
    on_topic = run(brain_env, "brain_evidence", {
        "question": "quokka nasturtium marsupial",
        "max_docs": 1,
    })
    off_topic = run(brain_env, "brain_evidence", {
        "question": "renormalization group fixed points in quantum field theory",
        "max_docs": 1,
    })

    on_evidence = on_topic.get("evidence", [])
    off_evidence = off_topic.get("evidence", [])
    assert on_evidence, f"on-topic query returned no evidence: {on_topic}"
    assert off_evidence, f"off-topic query returned no evidence: {off_topic}"

    on_score = on_evidence[0].get("score")
    off_score = off_evidence[0].get("score")
    assert isinstance(on_score, (int, float)), f"on-topic score not numeric: {on_score!r}"
    assert isinstance(off_score, (int, float)), f"off-topic score not numeric: {off_score!r}"

    # On-topic must clearly beat off-topic. A gap of >=0.15 in cosine-similarity
    # space is a conservative bar for the bge-small-en-v1.5 embedder against the
    # seed corpus; a regressed/near-random embedder collapses this gap toward 0.
    gap = on_score - off_score
    assert gap >= 0.15, (
        f"off-topic top score not meaningfully lower than on-topic: "
        f"on={on_score:.4f} off={off_score:.4f} gap={gap:.4f}"
    )


def test_evidence_max_docs_zero(brain_env):
    """max_docs=0 must return empty evidence, not crash or silently default."""
    out, is_error = run_raw(brain_env, "brain_evidence", {
        "question": "quokka nasturtium",
        "max_docs": 0,
    })
    assert not is_error, f"brain_evidence errored on max_docs=0: {out}"
    assert isinstance(out, dict), f"non-dict response: {out!r}"
    assert out.get("evidence") == [], f"expected empty evidence, got {out.get('evidence')!r}"


def test_evidence_max_docs_exceeds_corpus(brain_env):
    """max_docs > corpus size must cap at corpus size, not crash or duplicate."""
    out = run(brain_env, "brain_evidence", {
        "question": "quokka walrus",
        "max_docs": 999,
    })
    evidence = out.get("evidence", [])
    # Corpus has 2 seed docs. Should not return >2 entries or duplicates.
    paths = [e["path"] for e in evidence]
    assert len(paths) <= 2, f"returned more evidence than corpus has: {paths}"
    assert len(paths) == len(set(paths)), f"duplicate paths in evidence: {paths}"


def test_evidence_citations_point_at_real_files(brain_env):
    """Every cited path must exist inside the brain repo (no phantom citations)."""
    out = run(brain_env, "brain_evidence", {
        "question": "quokka nasturtium",
        "max_docs": 5,
    })
    repo = Path(brain_env["BRAIN_REPO"])
    for entry in out.get("evidence", []):
        full = repo / entry["path"]
        assert full.exists(), f"citation points at nonexistent file: {entry['path']}"
        # And must live under docs/, not .git/ or repo root.
        assert entry["path"].startswith("docs/"), \
            f"citation leaks path outside docs/: {entry['path']}"


def test_evidence_does_not_leak_git_internals(brain_env):
    """No citation may reference .git/ even if a query somehow scores it."""
    out = run(brain_env, "brain_evidence", {
        "question": "HEAD config refs objects",
        "max_docs": 10,
    })
    for entry in out.get("evidence", []):
        assert ".git" not in entry["path"], f".git leak: {entry['path']}"


def test_evidence_question_with_json_breaking_chars(brain_env):
    """Quotes and newlines in the question must not break JSON serialisation."""
    nasty = 'quokka "with quotes"\nand newline\tand tab \\ backslash'
    out, is_error = run_raw(brain_env, "brain_evidence", {
        "question": nasty,
        "max_docs": 2,
    })
    assert not is_error, f"evidence errored on nasty input: {out}"
    assert isinstance(out, dict), f"response not parseable JSON dict: {out!r}"
    assert out.get("question") == nasty, \
        f"question was mangled in echo: {out.get('question')!r} vs {nasty!r}"


def test_evidence_appends_exactly_one_retrieval_log_line(brain_env):
    """Telemetry consistency: brain_evidence must log like brain_search does.

    Bug hunt: brain_evidence calls brain_search internally (which logs once),
    then logs again itself. Does it double-count, or is it consistent?
    """
    before = len(_retrieval_log_lines(brain_env))
    run(brain_env, "brain_evidence", {"question": "quokka", "max_docs": 2})
    after = _retrieval_log_lines(brain_env)
    new_lines = after[before:]
    # There must be at least one "evidence_bundle" / kind=answer line.
    answer_lines = [l for l in new_lines if l.get("kind") == "answer"]
    assert len(answer_lines) == 1, \
        f"expected exactly 1 answer-kind log line, got {len(answer_lines)}: {new_lines}"


# ─────────────────────────── brain_audit ───────────────────────────


def test_audit_healthy_seed_reports_clean_and_counts(brain_env):
    """On the seed repo: clean=True, docs_total>=2, retrieval log path set."""
    out = run(brain_env, "brain_audit", {})
    assert out.get("clean") is True, f"seed repo flagged dirty: {out.get('dirty_status')!r}"
    assert out.get("docs_total", 0) >= 2, f"docs_total too low: {out}"
    assert out.get("retrieval_log"), "retrieval_log path missing from audit"
    vi = out.get("vector_index", {})
    assert vi.get("exists") is True, f"vector index should exist post-fixture: {vi}"


def test_audit_empty_docs_directory(brain_env):
    """Deleting every doc must not crash audit. docs_total should be 0."""
    repo = Path(brain_env["BRAIN_REPO"])
    docs = repo / "docs"
    for md in docs.glob("*.md"):
        md.unlink()
    out, is_error = run_raw(brain_env, "brain_audit", {})
    assert not is_error, f"audit errored on empty docs/: {out}"
    assert isinstance(out, dict), f"non-JSON response: {out!r}"
    assert out.get("docs_total") == 0, f"expected docs_total=0, got {out.get('docs_total')}"


def test_audit_dirty_working_tree(brain_env):
    """Uncommitted changes must surface as clean=False with non-empty dirty_status."""
    repo = Path(brain_env["BRAIN_REPO"])
    (repo / "docs" / "dirty.md").write_text("uncommitted\n")
    out = run(brain_env, "brain_audit", {})
    assert out.get("clean") is False, f"dirty tree reported clean: {out.get('dirty_status')!r}"
    assert out.get("dirty_status"), "dirty_status should be non-empty"


def test_audit_non_git_repo(brain_env):
    """Removing .git/ should produce a structured error, not a stack trace."""
    repo = Path(brain_env["BRAIN_REPO"])
    shutil.rmtree(repo / ".git")
    out, is_error = run_raw(brain_env, "brain_audit", {})
    # Acceptable: either is_error=True with text, OR JSON with an "error" key.
    if is_error:
        assert isinstance(out, (str, dict)), "errored response should carry a message"
    else:
        assert isinstance(out, dict), f"non-dict response: {out!r}"
        # If it didn't error, it should at least signal the broken git state.
        assert "error" in out or out.get("clean") is False, \
            f"missing .git/ silently passed audit: {out}"


def test_audit_calls_ensure_repo(brain_env):
    """Pointing BRAIN_REPO at a non-existent path must fail fast via _ensure_repo."""
    env = dict(brain_env)
    env["BRAIN_REPO"] = str(Path(brain_env["BRAIN_REPO"]).parent / "nonexistent")
    out, is_error = run_raw(env, "brain_audit", {})
    # Either MCP-level error or a JSON error payload. Must NOT silently succeed.
    assert is_error or (isinstance(out, dict) and "error" in out), \
        f"audit didn't reject nonexistent BRAIN_REPO: is_error={is_error} out={out!r}"
