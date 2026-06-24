"""Background index builder.

Run as `brein index build` (foreground, used by CI and tests) or
`brein index spawn` (detached, used by brain_resume_index and by
setup at the end of init). The worker:

  1. Marks state as `building` with its own pid.
  2. Calls vector._load_vector_index(force_rebuild=True), passing a
     heartbeat callback so progress lands in state.json every ~5s.
  3. On success: state goes to `ready` (and is then cleared, since
     resolve_status() can derive 'ready' from the index file alone).
  4. On failure: state goes to `stalled` with last_error set.

Detachment: spawn_detached() uses Popen with start_new_session and
fully redirected stdio so the worker survives the parent exiting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import index_state


def run() -> int:
    """Foreground build. Returns 0 on success, 1 on failure."""
    from . import vector

    index_state.update(status="building", worker_pid=os.getpid(), last_error="")
    last_tick = [0.0]

    def cb(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last_tick[0] >= 5.0 or done == total:
            last_tick[0] = now
            index_state.heartbeat(done=done, total=total)

    try:
        idx = vector._load_vector_index(progress_cb=cb, force_rebuild=True)
    except Exception as exc:
        index_state.update(
            status="stalled",
            last_error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        print(f"index build failed: {exc}", file=sys.stderr)
        return 1

    n = len(idx.get("entries", []))
    index_state.update(status="ready", done=n, total=n, last_error="")
    # Once the index file is on disk, resolve_status() can derive 'ready'
    # without us — keep state.json around for one cycle so callers see the
    # transition, then drop it on the next clean build.
    return 0


def spawn_detached() -> int:
    """Launch `brein index build` as a detached background process.

    Returns the worker pid. Does not wait. Caller should write the pid
    into state.json (the worker also does this on its own first tick,
    so a race is harmless).
    """
    brein = _brein_executable()
    log_path = _worker_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [brein, "index", "build"],
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from parent's process group
        close_fds=True,
    )
    # Pre-seed state so resolve_status() sees 'building' immediately
    # (worker's own first heartbeat will overwrite this within seconds).
    index_state.update(status="building", worker_pid=proc.pid, last_error="")
    return proc.pid


def _brein_executable() -> str:
    """Resolve the `brein` CLI to invoke. Prefer the same one currently running."""
    import shutil

    cand = sys.argv[0] if sys.argv and sys.argv[0] else None
    if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
        return cand
    return shutil.which("brein") or "brein"


def _worker_log_path() -> Path:
    return index_state.STATE_PATH.with_name("index-worker.log")
