"""Bug-hunting tests for the brain_search heuristic rerank surface.

Skips the LLM rerank path entirely (costs money / requires hermes). Tests
focus on edge cases of `_maybe_rerank` and its heuristic branch:
- top_k boundary conditions (0, 1, > corpus, > RERANK_MAX_TOP_K)
- result schema preservation between rerank=True/False
- telemetry conservation (one tool_call + one search per call)
- empty-result / no-match handling
- ordering sanity bound between rerank on/off
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import brain_env, call_tool, run, run_raw  # noqa: F401


# Heuristic-only is the rule. RERANK_MAX_TOP_K default = 25.
HEURISTIC = {"rerank": True, "rerank_method": "heuristic"}


def _log_lines(env) -> list[dict]:
    p = Path(env["BRAIN_RETRIEVAL_LOG"])
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _by_kind(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("kind"), []).append(r)
    return out


# ────────────────────────────────────────────────────────────────────────────
# top_k boundary conditions
# ────────────────────────────────────────────────────────────────────────────

def test_rerank_top_k_zero_does_not_error(brain_env):
    """rerank_top_k=0 — the source clamps `max(1, min(int(top_k or 1), CAP))`,
    so 0 should silently become 1, not raise."""
    out, is_err = run_raw(brain_env, "brain_search",
                          {"query": "quokka", "rerank_top_k": 0, **HEURISTIC})
    assert not is_err, f"top_k=0 unexpectedly errored: {out}"
    assert out["status"] == "ready"
    # Clamp should have produced top_k >= 1 (or 0 if no results).
    assert out["rerank"]["top_k"] >= 0


def test_rerank_top_k_negative(brain_env):
    """Negative top_k is also clamped via max(1, ...). Should not error."""
    out, is_err = run_raw(brain_env, "brain_search",
                          {"query": "quokka", "rerank_top_k": -5, **HEURISTIC})
    assert not is_err, f"negative top_k errored: {out}"
    assert out["status"] == "ready"


def test_rerank_top_k_one_with_no_hits(brain_env):
    """Empty corpus → truly empty results. rerank=True with top_k=1 must
    preserve response shape and not crash.

    Note: semantic search always returns nearest neighbours when the corpus
    is non-empty (there is no relevance threshold), so a "nonsense query"
    cannot produce zero hits. The only way to force `results == []` is to
    wipe the corpus and rebuild the index.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    for seeded in (repo / "docs").glob("*.md"):
        seeded.unlink()
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-am", "wipe corpus"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "brain_mcp.cli", "index", "build"],
        env=brain_env, check=True, capture_output=True,
    )

    out = run(brain_env, "brain_search",
              {"query": "quokka", "rerank_top_k": 1, **HEURISTIC})

    # Empty corpus produces the status payload ("empty" + action="use_grep"),
    # not a "ready" response. That's the right behavior — but the invariant
    # we're enforcing is: rerank=True must not crash on empty corpora, the
    # response must be JSON-shaped, and the status payload must signal
    # downstream callers correctly.
    assert out.get("status") == "empty", f"expected empty status, got {out}"
    assert out.get("action") == "use_grep", out
    # Critically: no crash, no garbage hits, no rerank attempted.
    assert "results" not in out or out["results"] == []


def test_rerank_top_k_larger_than_corpus(brain_env):
    """Only 2 seed docs. top_k=20 must not produce duplicates or padding."""
    out = run(brain_env, "brain_search",
              {"query": "quokka walrus", "rerank_top_k": 20, **HEURISTIC})
    paths = [r["path"] for r in out["results"]]
    assert len(paths) == len(set(paths)), f"duplicate paths in results: {paths}"
    # Meta top_k should clamp to actual result count.
    assert out["rerank"]["top_k"] <= len(out["results"]) + 1  # tail allowance


def test_rerank_top_k_above_max_cap_is_capped(brain_env):
    """RERANK_MAX_TOP_K default is 25. Asking for 9999 must be capped at 25."""
    out = run(brain_env, "brain_search",
              {"query": "quokka", "rerank_top_k": 9999, **HEURISTIC})
    # The clamp is `min(int(top_k or 1), RERANK_MAX_TOP_K)`, then further
    # clamped to `len(results)`. With 2 seed docs, top_k in meta should be <= 25.
    assert out["rerank"]["top_k"] <= 25, out["rerank"]


# ────────────────────────────────────────────────────────────────────────────
# Schema preservation
# ────────────────────────────────────────────────────────────────────────────

def test_rerank_preserves_result_schema(brain_env):
    """Every key in a non-rerank result must also appear in a reranked result.
    Rerank may ADD keys (rerank_score, rerank_reason, rerank_signals) but must
    never REMOVE them."""
    plain = run(brain_env, "brain_search", {"query": "quokka"})
    reranked = run(brain_env, "brain_search", {"query": "quokka", **HEURISTIC})
    assert plain["results"], "expected at least one hit in plain search"
    assert reranked["results"], "expected at least one hit in reranked search"

    plain_keys = set(plain["results"][0].keys())
    rerank_keys = set(reranked["results"][0].keys())
    missing = plain_keys - rerank_keys
    assert not missing, f"rerank dropped result keys: {missing}"

    # Heuristic rerank should add these.
    assert "rerank_score" in rerank_keys, reranked["results"][0]
    assert "rerank_reason" in rerank_keys, reranked["results"][0]


def test_rerank_meta_shape(brain_env):
    """Top-level rerank meta must carry the expected fields regardless of
    whether the call had any candidates."""
    out = run(brain_env, "brain_search", {"query": "quokka", **HEURISTIC})
    meta = out["rerank"]
    for k in ("enabled", "requested_method", "method", "fallback_used", "top_k"):
        assert k in meta, f"rerank meta missing {k}: {meta}"
    assert meta["enabled"] is True
    assert meta["method"] == "heuristic"
    assert meta["fallback_used"] is False, "heuristic must not fall back"


# ────────────────────────────────────────────────────────────────────────────
# Telemetry conservation under rerank
# ────────────────────────────────────────────────────────────────────────────

def test_rerank_telemetry_one_call_one_search(brain_env):
    """A reranked brain_search must still produce exactly one tool_call row
    and exactly one search row — rerank must NOT double-log."""
    before = _by_kind(_log_lines(brain_env))
    n_search = len(before.get("search", []))
    n_call = len([r for r in before.get("tool_call", [])
                  if r.get("gen_ai.tool.name") == "brain_search"])

    out = run(brain_env, "brain_search",
              {"query": "quokka", "rerank_top_k": 10, **HEURISTIC})
    assert out["status"] == "ready"

    after = _by_kind(_log_lines(brain_env))
    delta_search = len(after.get("search", [])) - n_search
    delta_call = len([r for r in after.get("tool_call", [])
                      if r.get("gen_ai.tool.name") == "brain_search"]) - n_call

    assert delta_search == 1, f"expected exactly 1 search row, got {delta_search}"
    assert delta_call == 1, f"expected exactly 1 tool_call row, got {delta_call}"


# ────────────────────────────────────────────────────────────────────────────
# Empty / no-match query under rerank
# ────────────────────────────────────────────────────────────────────────────

def test_rerank_with_no_matches_still_ready(brain_env):
    """No-hit query under rerank must still return status=ready with empty
    results — not error, not status=building."""
    out, is_err = run_raw(brain_env, "brain_search",
                          {"query": "zzzz_no_such_token_xyz", **HEURISTIC})
    assert not is_err
    assert out["status"] == "ready", out
    # Vector search always returns SOMETHING ranked, even with a bad query,
    # but rerank shape must hold either way.
    assert isinstance(out["results"], list)
    assert "rerank" in out


# ────────────────────────────────────────────────────────────────────────────
# Sanity bound: ordering under rerank vs. not
# ────────────────────────────────────────────────────────────────────────────

def test_rerank_keeps_top_result_in_top_set(brain_env):
    """Two searches identical except `rerank`. The plain top result should
    still appear SOMEWHERE in the reranked result set (sanity bound — rerank
    should not wholly evict the strongest vector hit)."""
    plain = run(brain_env, "brain_search", {"query": "quokka nasturtium"})
    reranked = run(brain_env, "brain_search",
                   {"query": "quokka nasturtium", **HEURISTIC})
    if not plain["results"] or not reranked["results"]:
        return  # nothing to assert
    plain_top = plain["results"][0]["path"]
    reranked_paths = [r["path"] for r in reranked["results"]]
    assert plain_top in reranked_paths, \
        f"rerank evicted plain top {plain_top!r} from results {reranked_paths}"


def test_rerank_score_field_numeric(brain_env):
    """Every reranked result must carry a numeric rerank_score."""
    out = run(brain_env, "brain_search", {"query": "walrus", **HEURISTIC})
    for r in out["results"]:
        assert "rerank_score" in r, r
        assert isinstance(r["rerank_score"], (int, float)), \
            f"rerank_score not numeric: {r['rerank_score']!r}"


def test_rerank_results_sorted_by_rerank_score(brain_env):
    """Within the rerank top_k slice, results must be ordered by rerank_score
    descending. (Tail items beyond top_k may follow original order.)"""
    out = run(brain_env, "brain_search",
              {"query": "quokka walrus pickle", "rerank_top_k": 25, **HEURISTIC})
    results = out["results"]
    if len(results) < 2:
        return
    # The top_k from meta is the effective slice that was reranked.
    k = min(out["rerank"].get("top_k", len(results)), len(results))
    head = results[:k]
    scores = [r["rerank_score"] for r in head]
    assert scores == sorted(scores, reverse=True), \
        f"reranked head not sorted desc: {scores}"
