"""Bug-hunting tests for git divergence / pull-FF / push failure paths.

Surface under test:
- `_pull_ff`: pull --ff-only — what about divergence?
- `_commit_push`: push is fire-and-forget on a background thread. Does
  `brain_update` report success when the push later fails?
- Network failures (origin URL points at a non-existent path).
- Recovery after a failed pull/push.
- Two clones racing brain_update on a shared remote.

No happy-paths. Every assertion is hunting a bug.
"""
from __future__ import annotations

import copy
import os
import subprocess
import time
from pathlib import Path

import pytest

from conftest import brain_env, call_tool, make_frontmatter, run, run_raw


# ── helpers ─────────────────────────────────────────────────────────────────

def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _repo(env: dict) -> Path:
    return Path(env["BRAIN_REPO"])


def _bare(env: dict) -> Path:
    # conftest names it tmp_path/brain.git, sibling of the working repo.
    return _repo(env).parent / "brain.git"


def _make_doc_body(title: str, marker: str) -> str:
    return make_frontmatter(title, ["divergence-test"]) + f"Marker: {marker}.\n"


def _push_divergent_commit(bare: Path, tmp_root: Path, marker: str) -> str:
    """Clone the bare, add a doc with `marker`, push back. Returns committed sha."""
    clone = tmp_root / f"clone-{marker}"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(clone)],
        check=True, capture_output=True,
    )
    _git(clone, "config", "user.email", "diverger@brein.sh")
    _git(clone, "config", "user.name", "Diverger")
    doc = clone / "docs" / f"divergent-{marker}.md"
    doc.write_text(_make_doc_body(f"Divergent {marker}", marker))
    _git(clone, "add", "-A")
    _git(clone, "commit", "-q", "-m", f"divergent commit {marker}")
    sha = _git(clone, "rev-parse", "HEAD").stdout.strip()
    _git(clone, "push", "-q", "origin", "main")
    return sha


# ── 1. Divergence: remote moved forward; local has nothing new ──────────────

def test_brain_update_with_remote_ahead_only(brain_env, tmp_path):
    """Remote has new commits we don't have. _pull_ff should silently catch us
    up (this is a real fast-forward). After that, our brain_update must succeed
    AND the divergent doc must be visible locally."""
    bare = _bare(brain_env)
    sha = _push_divergent_commit(bare, tmp_path, "ahead")

    payload, is_error = run_raw(brain_env, "brain_update", {
        "file_path": "docs/after-ahead.md",
        "content": _make_doc_body("After ahead", "after-ahead"),
        "commit_message": "after remote-ahead",
    })
    # BUG HUNT: does the write succeed cleanly when remote was ahead?
    assert not is_error, f"brain_update errored when remote was strictly ahead: {payload}"
    assert isinstance(payload, dict) and "error" not in payload, payload

    # The divergent doc must now exist locally (proof FF actually ran).
    assert (_repo(brain_env) / "docs" / "divergent-ahead.md").exists(), \
        "fast-forward did not bring remote commit into local working tree"

    # And HEAD must contain the remote sha as an ancestor.
    head = _git(_repo(brain_env), "log", "--format=%H").stdout
    assert sha in head, f"local history is missing the remote sha {sha}"


# ── 2. True divergence: local has a commit remote doesn't, and vice versa ───

def test_brain_update_when_local_and_remote_diverged(brain_env, tmp_path):
    """Local makes a commit. Remote independently makes a different commit.
    `_pull_ff` cannot fast-forward. brain_update must fail loudly — not
    silently swallow it, not commit on top of an inconsistent base."""
    repo = _repo(brain_env)
    bare = _bare(brain_env)

    # Local commit not yet on remote.
    local_doc = repo / "docs" / "local-only.md"
    local_doc.write_text(_make_doc_body("Local only", "local-only"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local-only commit")
    # Deliberately DO NOT push.

    # Remote independently advances.
    _push_divergent_commit(bare, tmp_path, "diverge")

    payload, is_error = run_raw(brain_env, "brain_update", {
        "file_path": "docs/while-diverged.md",
        "content": _make_doc_body("While diverged", "while-diverged"),
        "commit_message": "write while diverged",
    })

    # BUG HUNT: this must NOT silently succeed. Pull-ff will fail; the
    # rollback path or the error must surface to the caller.
    if not is_error and isinstance(payload, dict) and "error" not in payload:
        pytest.fail(
            f"brain_update returned success while local/remote were diverged: {payload}"
        )

    # Working tree must not have the new file committed on top of a stale base.
    # If the file exists but wasn't part of a successful commit, it should have
    # been rolled back.
    status = _git(repo, "status", "--short").stdout
    assert "while-diverged.md" not in status or "error" in (payload if isinstance(payload, dict) else {}), \
        f"file left dirty after failed update: status={status!r} payload={payload!r}"


# ── 3. Recovery: after a divergence failure, can we recover? ─────────────────

def test_recovery_after_diverged_failure(brain_env, tmp_path):
    """After a diverged failure, manually reconciling (merge/rebase) should
    allow the next brain_update to succeed. If state is permanently broken,
    that's a bug."""
    repo = _repo(brain_env)
    bare = _bare(brain_env)

    # Create divergence.
    local_doc = repo / "docs" / "local-only-2.md"
    local_doc.write_text(_make_doc_body("Local only 2", "local-only-2"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local commit")
    _push_divergent_commit(bare, tmp_path, "recover")

    # First call: expected to fail or roll back.
    run_raw(brain_env, "brain_update", {
        "file_path": "docs/recover-attempt.md",
        "content": _make_doc_body("Recover attempt", "recover-attempt"),
        "commit_message": "attempt during divergence",
    })

    # Reconcile manually: rebase local onto remote.
    rebase = _git(repo, "pull", "--rebase", "origin", "main", check=False)
    assert rebase.returncode == 0, f"manual rebase failed: {rebase.stderr}"
    _git(repo, "push", "-q", "origin", "main")

    # Now a fresh brain_update should work cleanly.
    payload, is_error = run_raw(brain_env, "brain_update", {
        "file_path": "docs/post-recovery.md",
        "content": _make_doc_body("Post recovery", "post-recovery"),
        "commit_message": "after recovery",
    })
    assert not is_error and isinstance(payload, dict) and "error" not in payload, \
        f"brain_update broken even after manual reconciliation: {payload}"


# ── 4. Push failure: origin points at a non-existent path ────────────────────

def test_brain_update_when_origin_unreachable(brain_env):
    """Repoint origin at a path that doesn't exist. _pull_ff will fail.
    brain_update must NOT report success."""
    repo = _repo(brain_env)
    bogus = repo.parent / "does-not-exist.git"
    _git(repo, "remote", "set-url", "origin", str(bogus))

    payload, is_error = run_raw(brain_env, "brain_update", {
        "file_path": "docs/unreachable.md",
        "content": _make_doc_body("Unreachable", "unreachable"),
        "commit_message": "with unreachable origin",
    })

    # BUG HUNT: with origin broken, pull --ff-only must fail. Server should
    # error or report an error in the payload — NOT pretend it worked.
    if not is_error and isinstance(payload, dict) and "error" not in payload:
        pytest.fail(
            f"brain_update returned success with unreachable origin: {payload}"
        )


# ── 5. Push failure AFTER pull succeeds: origin breaks between pull & push ──

def test_push_failure_silently_succeeds(brain_env):
    """The known smell: _commit_push fires push on a background thread and
    returns {'pushed': 'pending'}. If the push later fails, the user already
    got a success response. This test makes the push impossible by breaking
    origin AFTER the pull would have happened.

    Strategy: pull succeeds (origin still valid for pull because pull happens
    before we break it). To break only the push, we set origin to a bogus
    URL via post-receive on the bare? Simpler: break origin AFTER call;
    instead break BOTH but the test for divergent push silently-succeeding
    is captured here by breaking origin entirely and observing payload.

    Because pull runs first and will fail when origin is bogus, this case
    is functionally similar to test_brain_update_when_origin_unreachable.
    Instead we use a remote that ACCEPTS pull (it exists, no new commits)
    but REJECTS push. We do that with a pre-receive hook in the bare repo
    that exits non-zero.
    """
    repo = _repo(brain_env)
    bare = _bare(brain_env)
    hooks = bare / "hooks"
    hooks.mkdir(exist_ok=True)
    pre_receive = hooks / "pre-receive"
    pre_receive.write_text("#!/bin/sh\necho 'rejected by hook' >&2\nexit 1\n")
    pre_receive.chmod(0o755)

    payload, is_error = run_raw(brain_env, "brain_update", {
        "file_path": "docs/push-rejected.md",
        "content": _make_doc_body("Push rejected", "push-rejected"),
        "commit_message": "push will be rejected",
    })

    # BUG HUNT: the response will almost certainly claim success because push
    # is backgrounded. Surface that.
    if isinstance(payload, dict) and payload.get("pushed") == "pending" and "error" not in payload:
        # Give the background push time to fail.
        time.sleep(1.0)
        # The bare repo must not have received the new commit.
        bare_log = subprocess.run(
            ["git", "-C", str(bare), "log", "--oneline", "main"],
            capture_output=True, text=True,
        ).stdout
        assert "push will be rejected" not in bare_log, \
            "bare somehow received a commit it was supposed to reject"

        # Local committed something the remote never got. brain_update lied
        # to the caller by returning a success-shaped response.
        pytest.fail(
            "brain_update returned pushed='pending' but the remote rejected "
            "the push — caller has no way to discover that. "
            f"payload={payload}"
        )


# ── 6. After a failed push, next brain_update sees fresh state? ──────────────

def test_next_update_after_silent_push_failure(brain_env):
    """If the first push silently failed (hook rejection), a SECOND
    brain_update without a remote update should still work locally — but
    if the remote rejects again, we accumulate unpushed commits with no
    signal. Verify how the second call behaves."""
    repo = _repo(brain_env)
    bare = _bare(brain_env)

    # Install rejecting hook.
    hooks = bare / "hooks"
    hooks.mkdir(exist_ok=True)
    pre_receive = hooks / "pre-receive"
    pre_receive.write_text("#!/bin/sh\nexit 1\n")
    pre_receive.chmod(0o755)

    run_raw(brain_env, "brain_update", {
        "file_path": "docs/first-rejected.md",
        "content": _make_doc_body("First rejected", "first-rejected"),
        "commit_message": "first (will be rejected)",
    })
    time.sleep(0.5)

    payload2, is_error2 = run_raw(brain_env, "brain_update", {
        "file_path": "docs/second-rejected.md",
        "content": _make_doc_body("Second rejected", "second-rejected"),
        "commit_message": "second (will be rejected)",
    })
    time.sleep(0.5)

    # Count unpushed commits.
    unpushed = _git(repo, "log", "origin/main..HEAD", "--oneline").stdout.strip().splitlines()

    # BUG HUNT: brain_update accumulates unpushed commits silently. If both
    # writes "succeeded" yet remote rejected both, we have 2 unpushed commits
    # and zero error surface to the caller.
    # After the sync-push fix, a rejected push surfaces as pushed != "ok"
    # (typically "failed") plus a push_error field. "Success" from the
    # caller's POV is pushed == "ok".
    p2_success = (
        not is_error2
        and isinstance(payload2, dict)
        and payload2.get("pushed") == "ok"
        and payload2.get("changed") is True
    )
    if p2_success and len(unpushed) >= 2:
        pytest.fail(
            f"two brain_updates returned pushed='ok' but {len(unpushed)} commits "
            "are stuck locally; remote was rejecting both"
        )


# ── 7. Two clones racing the same remote ─────────────────────────────────────

def test_two_writers_second_must_see_first(brain_env, tmp_path):
    """Simulate two MCP servers (two clones) writing concurrently. The
    second writer's _pull_ff must pick up the first writer's commit, or
    its push will be rejected. Either is OK — but silent loss is a bug."""
    bare = _bare(brain_env)

    # Build a second clone with its own brain-mcp env.
    second_repo = tmp_path / "brain-second"
    second_home = tmp_path / "home-second"
    (second_home / ".brein").mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(second_repo)],
        check=True, capture_output=True,
    )
    _git(second_repo, "config", "user.email", "second@brein.sh")
    _git(second_repo, "config", "user.name", "Second")

    second_env = copy.deepcopy(brain_env)
    second_env["BRAIN_REPO"] = str(second_repo)
    second_env["BRAIN_RETRIEVAL_LOG"] = str(second_home / ".brein" / "retrieval-log.jsonl")
    second_env["BRAIN_VECTOR_INDEX"] = str(second_home / ".brein" / "vector-index.json")
    second_env["HOME"] = str(second_home)

    # Build index for second clone (mirrors conftest).
    subprocess.run(
        [os.environ.get("PYTHON", "python3"), "-m", "brain_mcp.cli", "index", "build"],
        env=second_env, check=False, capture_output=True,
    )

    # Writer 1 (the original env) writes and pushes.
    p1, e1 = run_raw(brain_env, "brain_update", {
        "file_path": "docs/writer-one.md",
        "content": _make_doc_body("Writer one", "writer-one"),
        "commit_message": "writer one commit",
    })
    assert not e1 and isinstance(p1, dict) and "error" not in p1, p1
    # Wait for background push.
    time.sleep(1.0)

    # Verify writer-one actually landed on the bare.
    bare_log = subprocess.run(
        ["git", "-C", str(bare), "log", "--oneline", "main"],
        capture_output=True, text=True,
    ).stdout
    assert "writer one commit" in bare_log, \
        f"writer-one's push did not reach bare: {bare_log}"

    # Writer 2 now writes. Its _pull_ff must pick up writer-one's commit.
    p2, e2 = run_raw(second_env, "brain_update", {
        "file_path": "docs/writer-two.md",
        "content": _make_doc_body("Writer two", "writer-two"),
        "commit_message": "writer two commit",
    })
    assert not e2 and isinstance(p2, dict) and "error" not in p2, p2
    time.sleep(1.0)

    # BUG HUNT: writer-two's local repo must now contain writer-one's file.
    # If not, writer-two committed on a stale base — silent data loss risk.
    assert (second_repo / "docs" / "writer-one.md").exists(), \
        "writer-two did not pull writer-one's commit before writing"

    # And the bare must have both commits in linear history.
    final_log = subprocess.run(
        ["git", "-C", str(bare), "log", "--oneline", "main"],
        capture_output=True, text=True,
    ).stdout
    assert "writer one commit" in final_log and "writer two commit" in final_log, \
        f"bare missing one of the two writers' commits: {final_log}"
