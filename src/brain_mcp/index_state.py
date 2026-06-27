"""Index build state: a tiny JSON file siblinged to vector-index.json.

The state file is the source of truth for "is the index ready, building,
or stuck". brain_search reads it before doing anything; the background
worker writes to it; brain_resume_index inspects + acts on it.

State machine:
    missing  -> no state file, no index file. Fresh install.
    empty    -> index file exists but has 0 entries. New/empty brain.
    building -> worker pid alive, heartbeat fresh.
    stalled  -> state says building, but worker dead or heartbeat stale.
    ready    -> index file exists, no active build.

Callers should use `resolve_status()` which verifies worker liveness
and heartbeat freshness rather than trusting the raw `status` field.
"""

from __future__ import annotations

import errno
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import VECTOR_INDEX_PATH

IndexStatus = Literal["missing", "empty", "building", "stalled", "ready"]

STATE_PATH = VECTOR_INDEX_PATH.with_name("index-state.json")
STALE_HEARTBEAT_SECONDS = 90  # worker writes every ~5s; 90s without = dead


@dataclass(frozen=True)
class IndexState:
    status: IndexStatus
    started_at: str
    updated_at: str
    worker_pid: int | None = None
    done: int = 0
    total: int = 0
    last_error: str | None = None
    build_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_state() -> IndexState | None:
    if not STATE_PATH.exists():
        return None
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return IndexState(**raw)
    except TypeError:
        return None


def write_state(state: IndexState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: write to temp + rename. Avoids torn reads if a brain_search
    # is racing the worker.
    fd, tmp = tempfile.mkstemp(
        dir=str(STATE_PATH.parent), prefix=".index-state.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state.to_json(), fh, indent=2)
        os.replace(tmp, STATE_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update(
    *,
    status: IndexStatus | None = None,
    done: int | None = None,
    total: int | None = None,
    worker_pid: int | None = None,
    last_error: str | None = None,
) -> IndexState:
    """Partial update. Bumps `updated_at` (heartbeat) every call."""
    prev = read_state()
    now = _now_iso()
    base = prev or IndexState(status="building", started_at=now, updated_at=now)
    merged = IndexState(
        status=status if status is not None else base.status,
        started_at=base.started_at if prev else now,
        updated_at=now,
        worker_pid=worker_pid if worker_pid is not None else base.worker_pid,
        done=done if done is not None else base.done,
        total=total if total is not None else base.total,
        last_error=last_error if last_error is not None else base.last_error,
        build_id=base.build_id,
    )
    write_state(merged)
    return merged


def heartbeat(done: int, total: int) -> IndexState:
    """Shortcut for the worker's tight loop: bump progress + updated_at."""
    return update(done=done, total=total)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # alive but owned by another user
    return True


def _heartbeat_age_seconds(state: IndexState) -> float:
    try:
        dt = datetime.fromisoformat(state.updated_at)
    except ValueError:
        return float("inf")
    return max(0.0, time.time() - dt.timestamp())


def resolve_status() -> tuple[IndexStatus, IndexState | None]:
    """Compute the *actual* status, verifying worker liveness.

    Don't trust state.status blindly — a worker can be killed mid-build,
    leaving status="building" forever. We verify with the pid + heartbeat.
    """
    state = read_state()
    index_exists = VECTOR_INDEX_PATH.exists()

    if state is None:
        if not index_exists:
            return ("missing", None)
        return ("ready", None) if _index_has_entries() else ("empty", None)

    if state.status == "building":
        alive = _pid_alive(state.worker_pid)
        fresh = _heartbeat_age_seconds(state) < STALE_HEARTBEAT_SECONDS
        if alive and fresh:
            return ("building", state)
        return ("stalled", state)

    if state.status == "ready":
        if not index_exists:
            return ("missing", state)
        return ("ready" if _index_has_entries() else "empty", state)

    return (state.status, state)


def _index_has_entries() -> bool:
    if not VECTOR_INDEX_PATH.exists():
        return False
    try:
        raw = json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(raw, dict):
        return False
    return bool(raw.get("entries"))


def clear() -> None:
    """Remove the state file. Used after a successful build or on reset."""
    try:
        STATE_PATH.unlink()
    except FileNotFoundError:
        pass
