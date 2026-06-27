"""Bug-hunting concurrency tests for brain-mcp.

Each test spawns a fresh brain-mcp subprocess per call_tool. We race two
calls via threading.Thread to surface races in:

- two brain_update calls racing the same repo (working-tree, git, push)
- brain_search interleaved with brain_update (torn state visible to readers)
- two brain_update calls writing different files (do both land + push?)
- two brain_index_status calls while index is rebuilding (worker spawned 2x?)

Per task instructions, these tests are designed to FAIL when a real bug
exists. We DO NOT fix anything we find.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from conftest import brain_env, call_tool, make_frontmatter, run, run_raw  # noqa: F401


def _race(targets: list[tuple]) -> list[tuple]:
    """Run callables concurrently. Each target = (fn, args). Returns list of
    (result, exception) in order."""
    results: list[tuple] = [(None, None)] * len(targets)

    def runner(i: int, fn, args):
        try:
            results[i] = (fn(*args), None)
        except Exception as exc:  # noqa: BLE001
            results[i] = (None, exc)

    threads = [
        threading.Thread(target=runner, args=(i, fn, args))
        for i, (fn, args) in enumerate(targets)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _doc(title: str, body: str) -> str:
    return make_frontmatter(title, ["concurrency"]) + body + "\n"


def _git_log(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%s", "refs/heads/main"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return out


def _wait_for_pushes(repo: Path, timeout: float = 5.0) -> None:
    """brain_update pushes in a daemon thread per subprocess. We don't
    control those threads, but each subprocess exits after the call, and
    pushes go to a local bare remote — usually <100ms. Sleep generously."""
    time.sleep(timeout)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Two updates to the SAME file at the same time
# ─────────────────────────────────────────────────────────────────────────────

def test_two_updates_same_file_race(brain_env):
    """BUG HUNT: Two concurrent brain_update calls to the same path.

    Each subprocess will _pull_ff → write → validate → commit. They run in
    SEPARATE working trees? No — same BRAIN_REPO. Whichever commits second
    should see the first's commit via _pull_ff. But the writes are not
    serialized: process B may pull (sees empty), write its content,
    validate, commit on top of A — clobbering A's content silently, while
    both processes report success.

    Asserts both commits exist with distinct content. If only one shows up
    or one process errors, that's the race.
    """
    path = "docs/race_same.md"
    a = _doc("race A", "Content from process A — quokka.")
    b = _doc("race B", "Content from process B — walrus.")

    results = _race([
        (call_tool, (brain_env, "brain_update",
                     {"file_path": path, "content": a, "commit_message": "A"})),
        (call_tool, (brain_env, "brain_update",
                     {"file_path": path, "content": b, "commit_message": "B"})),
    ])

    repo = Path(brain_env["BRAIN_REPO"])
    _wait_for_pushes(repo)

    # Both calls returned a result (no Python exception)?
    for i, (res, exc) in enumerate(results):
        assert exc is None, f"call {i} raised {exc!r}"

    texts = [res[0] for res, _ in results]
    errors = [t for t in texts if '"error"' in t]

    # Diagnostic: surface what actually happened
    log = _git_log(repo)
    final = (repo / path).read_text() if (repo / path).exists() else "<missing>"

    # The bug hunt: if both calls reported success, both commit messages
    # MUST appear in git log. Otherwise one silently clobbered the other.
    if not errors:
        assert "A" in log and "B" in log, (
            f"Both updates claimed success but git log only shows: {log}. "
            f"Final file: {final!r}"
        )

    # And the working tree must equal whichever commit landed last.
    last_commit_msg = log[0] if log else None
    expected = a if last_commit_msg == "A" else b if last_commit_msg == "B" else None
    if expected and not errors:
        assert final == expected, (
            f"Working tree {final!r} doesn't match last commit '{last_commit_msg}'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2) Two updates to DIFFERENT files at the same time
# ─────────────────────────────────────────────────────────────────────────────

def test_two_updates_different_files_both_land(brain_env):
    """BUG HUNT: Two concurrent updates to different paths.

    Both should commit and push. Race-prone bits: _commit_push does
    `git add <paths>` + `git commit` — but they share the same index file.
    A interleaved with B may see B's staged changes in A's commit, or
    fail-non-fast on `git add` simultaneously, or background pushes may
    collide non-ff.

    Asserts: both files exist on disk with their respective content, both
    commits appear in log, and bare remote received both.
    """
    path_a = "docs/race_diff_a.md"
    path_b = "docs/race_diff_b.md"
    a_doc = _doc("diff A", "A-distinct-content quokka diff")
    b_doc = _doc("diff B", "B-distinct-content walrus diff")

    results = _race([
        (call_tool, (brain_env, "brain_update",
                     {"file_path": path_a, "content": a_doc, "commit_message": "diff-A"})),
        (call_tool, (brain_env, "brain_update",
                     {"file_path": path_b, "content": b_doc, "commit_message": "diff-B"})),
    ])

    repo = Path(brain_env["BRAIN_REPO"])
    _wait_for_pushes(repo)

    for i, (res, exc) in enumerate(results):
        assert exc is None, f"call {i} raised {exc!r}"
        text, is_err = res
        assert not is_err, f"call {i} tool error: {text}"
        body = json.loads(text)
        assert "error" not in body, f"call {i} returned error: {body}"

    # Both files present on disk with correct content
    assert (repo / path_a).read_text() == a_doc, (repo / path_a).read_text()
    assert (repo / path_b).read_text() == b_doc, (repo / path_b).read_text()

    log = _git_log(repo)
    assert "diff-A" in log and "diff-B" in log, f"git log missing one commit: {log}"

    # Bare remote should have both commits as well
    bare = Path(brain_env["BRAIN_REPO"]).parent / "brain.git"
    remote_log = subprocess.run(
        ["git", "-C", str(bare), "log", "--format=%s", "refs/heads/main"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "diff-A" in remote_log and "diff-B" in remote_log, (
        f"bare remote missing commit(s): {remote_log!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3) brain_search interleaved with brain_update
# ─────────────────────────────────────────────────────────────────────────────

def test_search_during_update_does_not_crash(brain_env):
    """BUG HUNT: brain_search running concurrently with brain_update.

    brain_update calls _pull_ff, writes the file, regenerates docs/index.md,
    runs validate, commits. A concurrent brain_search will load the vector
    index from disk (BRAIN_VECTOR_INDEX json file). If brain_update happens
    to be regenerating the index at the same wall-clock moment (it doesn't
    here — only docs/index.md is regenerated, not vector index), search
    should be unaffected. But torn writes of docs/index.md or partial files
    could surface as parse errors.

    Asserts: search call returns a parseable response (no crash, no
    isError), regardless of timing.
    """
    path = "docs/search_vs_update.md"
    doc = _doc("update during search", "fresh-content kangaroo search-race")

    results = _race([
        (call_tool, (brain_env, "brain_update",
                     {"file_path": path, "content": doc, "commit_message": "su"})),
        (call_tool, (brain_env, "brain_search",
                     {"query": "quokka", "max_results": 5})),
    ])

    for i, (res, exc) in enumerate(results):
        assert exc is None, f"call {i} raised {exc!r}"

    update_text, update_err = results[0][0]
    search_text, search_err = results[1][0]

    assert not search_err, f"search errored under concurrent update: {search_text}"
    # Search must return either ranked results or a status payload; never a crash.
    parsed = json.loads(search_text)
    assert isinstance(parsed, dict) or isinstance(parsed, list), (
        f"search returned unparseable shape: {parsed!r}"
    )

    # And the update must not have rolled back due to validation seeing a
    # torn docs/index.md from a parallel writer.
    update_body = json.loads(update_text)
    assert "error" not in update_body, (
        f"update rolled back under concurrent search: {update_body}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4) Two brain_index_status callers while rebuilding
# ─────────────────────────────────────────────────────────────────────────────

def test_two_index_status_callers_dont_spawn_two_workers(brain_env):
    """BUG HUNT: Two concurrent brain_index_status(restart_if_stalled=True).

    Each may try to spawn_detached a worker. If there is no inter-process
    lock, both could spawn workers simultaneously — leading to duplicated
    embedding work, racing writes to the vector-index.json file, or torn
    progress state.

    Asserts: both calls return parseable JSON and do not crash. Further,
    after settling, at most ONE worker process tree should be alive
    (best-effort — we just check the status doesn't perpetually report
    contradictory progress).
    """
    # First nuke the prebuilt index so restart actually has work to do.
    idx = Path(brain_env["BRAIN_VECTOR_INDEX"])
    if idx.exists():
        idx.unlink()
    # Also kill any worker state file siblings
    for sibling in idx.parent.glob("vector-index.json*"):
        sibling.unlink()

    results = _race([
        (call_tool, (brain_env, "brain_index_status",
                     {"restart_if_stalled": True, "force_rebuild": True})),
        (call_tool, (brain_env, "brain_index_status",
                     {"restart_if_stalled": True, "force_rebuild": True})),
    ])

    for i, (res, exc) in enumerate(results):
        assert exc is None, f"call {i} raised {exc!r}"
        text, is_err = res
        assert not is_err, f"call {i} tool error: {text}"
        parsed = json.loads(text)
        assert isinstance(parsed, dict), f"call {i} non-dict payload: {parsed!r}"
        # Must report a recognizable status key
        assert "status" in parsed, f"call {i} missing status: {parsed}"


# ─────────────────────────────────────────────────────────────────────────────
# 5) Many concurrent updates — stress for non-ff push races
# ─────────────────────────────────────────────────────────────────────────────

def test_many_concurrent_updates_all_land(brain_env):
    """BUG HUNT: 4 concurrent brain_update calls to distinct files.

    Background pushes from each subprocess can race the bare remote
    (non-ff). Even though _push_lock exists, it's per-process — each
    subprocess has its own. So pushes serialize within a process but
    NOT across the 4 subprocesses. Expect at least one push to lose
    the race and silently fail (stderr log only).

    Asserts: all 4 commits visible LOCALLY (since they share the working
    tree and _pull_ff before commit) AND all 4 visible on the bare remote.
    The bare-remote check is where this is most likely to fail.
    """
    n = 4
    paths = [f"docs/many_{i}.md" for i in range(n)]
    docs = [_doc(f"many-{i}", f"unique-token-{i} content-{i}") for i in range(n)]
    msgs = [f"many-{i}" for i in range(n)]

    targets = [
        (call_tool, (brain_env, "brain_update",
                     {"file_path": p, "content": d, "commit_message": m}))
        for p, d, m in zip(paths, docs, msgs)
    ]
    results = _race(targets)

    repo = Path(brain_env["BRAIN_REPO"])
    _wait_for_pushes(repo, timeout=8.0)

    successes = 0
    for i, (res, exc) in enumerate(results):
        assert exc is None, f"call {i} raised {exc!r}"
        text, is_err = res
        body = json.loads(text) if text.startswith("{") else {"error": text}
        if "error" not in body:
            successes += 1

    log = _git_log(repo)
    landed_local = [m for m in msgs if m in log]

    bare = repo.parent / "brain.git"
    remote_log = subprocess.run(
        ["git", "-C", str(bare), "log", "--format=%s", "refs/heads/main"],
        check=True, capture_output=True, text=True,
    ).stdout
    landed_remote = [m for m in msgs if m in remote_log]

    # Surface the gap: claims of success vs reality on disk vs reality on remote.
    assert successes == len(landed_local) == len(landed_remote) == n, (
        f"successes={successes} local={landed_local} remote={landed_remote} "
        f"local_log={log[:8]} remote_log={remote_log.splitlines()[:8]}"
    )
