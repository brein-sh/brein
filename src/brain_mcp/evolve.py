"""Self-improving brain loop.

After every N=50 ab_run rows (configurable via BRAIN_EVOLVE_EVERY), a
detached worker reads the recent no-brain wins from eval-log.jsonl and
asks an agentic LLM with Read/Grep/Glob/Edit tools to:

  1. Identify which brain doc was the canonical source for the question.
  2. Diff the brain_answer vs no_brain_answer; extract the concrete refs
     (file paths, line numbers, function names) the no-brain side had.
  3. Verify every ref via Grep against actual source. Drop unverifiable.
  4. Edit the brain doc to add a "## Source references" section so the
     next similar question wins.

All edits land in one commit + push under the interprocess write lock.
Each run logs a single row to ~/.brein/evolve-log.jsonl.

The 13/13 losses in the current eval log all share one pattern: no-brain
won on specificity (paths, line numbers). This loop closes the gap.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import REPO_PATH


EVAL_LOG_PATH = REPO_PATH / ".brain" / "eval-log.jsonl"
EVOLVE_LOG_PATH = Path(os.path.expanduser("~")) / ".brein" / "evolve-log.jsonl"
# Per-loss progress, written BEFORE and AFTER each loss so `tail -f` shows
# a live cursor while evolve is mid-run (one cycle can take 30+ minutes
# across 13 losses).
EVOLVE_PROGRESS_PATH = Path(os.path.expanduser("~")) / ".brein" / "evolve-progress.jsonl"
EVOLVE_TRIGGER_EVERY = int(os.environ.get("BRAIN_EVOLVE_EVERY", "50"))
EVOLVE_TIMEOUT_SECONDS = float(os.environ.get("BRAIN_EVOLVE_TIMEOUT_S", "900"))
# Per-loss agentic LLM calls are independent (different canonical doc each)
# and each takes 30-90s. Fan out across a thread pool — claude CLI calls
# are blocking subprocess.run, so threads (not processes) are right.
EVOLVE_PARALLELISM = int(os.environ.get("BRAIN_EVOLVE_PARALLELISM", "8"))
EVOLVE_GUARD_ENV = "BRAIN_EVOLVE_IN_PROGRESS"


EvolveKind = Literal["improved", "skipped", "escalated", "error", "noop"]


@dataclass(frozen=True)
class EvolveResult:
    evolve_id: str
    started_at: str
    losses_examined: int
    losses_improved: int
    losses_escalated: int
    losses_skipped: int
    commit_sha: str | None = None
    losses: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── eval-log readers ────────────────────────────────────────────────────
#
# Schema note: A/B verdict rows in eval-log.jsonl have NO `kind` field —
# they carry `verdict ∈ {brain_better, tie, no_brain_better}` at top level.
# Only `gate_skipped` rows have a `kind` field. Original v0.5.24 filtered
# on `kind == "ab_run"` and silently matched nothing. v0.5.26 fix: detect
# A/B rows by the presence of a verdict in the known set.

_AB_VERDICTS = {"brain_better", "tie", "no_brain_better"}


def _count_ab_runs() -> int:
    if not EVAL_LOG_PATH.exists():
        return 0
    n = 0
    for line in EVAL_LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("verdict") in _AB_VERDICTS:
            n += 1
    return n


def _read_recent_losses(limit: int = 50) -> list[dict[str, Any]]:
    """Last N A/B rows that carry an explicit `lesson` (trajectory gate fired).

    Old rows without `lesson` are intentionally ignored — they were judged
    by the pairwise "more specific" rubric, which is the rot-pump we
    replaced. Re-evaluating them under the new gate is fine; learning from
    them under the old verdict is not.
    """
    if not EVAL_LOG_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in EVAL_LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("lesson"):
            rows.append(d)
    return rows[-limit:]


# ── agentic prompt ─────────────────────────────────────────────────────

_EVOLVE_PROMPT = """You are the brain self-improvement agent for the company brain at:
{repo}

The brain is a MAP, not a manual. It points an agent at the right repo or
module; the agent re-derives line-level specifics by reading live source.
Line numbers, exact symbol names, and pasted function bodies ROT on the
next refactor — they have no place here.

An eval gate just identified a missing pointer. On the question below,
a no-brain agent (with grep/repo tools only) had to discover files that
the brain-on agent never opened. Those files are the brain's blind spot.

Your job: add a one-line, MODULE-LEVEL pointer to the right brain doc so
next time the brain-on agent goes there first.

You have these tools: Read, Grep, Glob, Edit. Use them.

Inputs:
  Topic            : {topic}
  Pointer files    : {pointer_files}
                     (these are the files no-brain converged on; treat
                     them as candidates, not as ground truth — verify
                     each one exists at its current path before pointing
                     to it)

Workflow:
  1. Verify each pointer file still exists (Glob / Read). Drop any that
     don't. If none survive, output kind="skipped".
  2. Reduce each pointer to its module — the directory or a short path
     that names the area, NOT the specific file. e.g. `services/auth/`
     not `services/auth/handlers/login.py`. A module rename is rare; a
     file-or-line change is constant.
  3. Find the canonical brain doc for this topic. Grep `docs/` for the
     topic keywords; Read the top candidates. Prefer docs with
     `source_of_truth: true` frontmatter. If a clearly-canonical doc
     exists, edit it. If NO doc covers this topic at all, create a new
     pointer doc at `docs/pointers/<slug>.md` (slug = kebab-case topic).
  4. Add or extend a section titled exactly `## Where to look`. Format
     each entry as:
        - `path/to/module/` — one short sentence: what lives here.
     ONE entry per module. Do not list individual files. Do not include
     line numbers, line ranges, function names, class names, or pasted
     code. Pointers, not facts.
  5. Preserve frontmatter exactly. Do not rewrite other sections.

Hard rules — violations will be rejected at commit time and the entire
edit reverted, wasting the cycle. Read carefully:
  - NO `:L<digits>` anywhere in your final edit (e.g. `foo.py:L42`).
  - NO `:<digits>` line refs (e.g. `foo.py:42`, `auth.ts:100-150`).
  - NO line ranges anywhere — "lines 100-200", "(lines 42-80)", "L42-L80".
  - NO pasted function bodies, no fenced code blocks containing source.
  - NO specific symbol names (function names, class names, variable names)
    inside `## Where to look` entries. Module/directory level only.
  - If the LESSON section below quotes brittle text from a prior answer,
    DO NOT preserve that quoting in your edit. The hard rules apply to
    the FINAL FILE CONTENT, not just to what you write — if the doc you
    edit already contains line numbers anywhere (left over from old rot),
    your edit will be reverted unless you ALSO scrub those lines as part
    of this change.
  - When in doubt, drop the entry. A missing pointer is recoverable; a
    reverted commit is wasted work.

Self-check before you call your final Edit:
  1. Grep your proposed content for `:L`, `.py:`, `.ts:`, `.js:`, `.md:`,
     `lines `, `line `. Any hit → rewrite without it.
  2. If the file already contains brittle lines outside your edit, either
     fix them in the same edit or output kind="escalated" with reason
     "preexisting_brittle_content".

Idempotency:
  - If a `## Where to look` entry for the same module already exists,
    leave it. Do not duplicate.
  - If an existing entry has line numbers, REWRITE it to module-level
    (this is a real improvement, not a violation).

Output ONE JSON object (no prose, no markdown fence):

{{
  "kind": "improved" | "skipped" | "escalated",
  "confidence": "high" | "medium" | "low",
  "canonical_path": "docs/..." or null,
  "pointers_added": ["services/auth/ — handles login", ...],
  "edits_applied": true or false,
  "summary": "one-line plain English",
  "escalation_reason": "..." or null
}}

--- LESSON ---
Question         : {question}
Reason gate fired: {reason}

Brain answer (existing context, for understanding what the brain already said):
{brain_answer}

No-brain answer (what the agent reached by exploring):
{no_brain_answer}
""".rstrip()


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


_BRITTLE_PATTERNS = (
    re.compile(r":L\d+"),                  # foo.py:L42
    re.compile(r"\.[a-zA-Z]{1,5}:\d+"),    # foo.py:42
    re.compile(r"\(lines?\s+\d+", re.IGNORECASE),
)


def _has_brittle_specifics(text: str) -> str | None:
    """Return the offending pattern if text contains brittle specifics."""
    for pat in _BRITTLE_PATTERNS:
        m = pat.search(text or "")
        if m:
            return m.group(0)
    return None


def _revert_brittle_edits(canonical_path: str | None) -> None:
    """If the LLM violated the no-line-numbers rule, drop its edit. Uses
    git checkout to restore the file (or remove it if it was new)."""
    if not canonical_path:
        return
    from .shared import _run_git
    rel = canonical_path
    # If the file is tracked, restore it; if not, it was new — delete it.
    ls = _run_git(["ls-files", "--error-unmatch", rel])
    if ls.returncode == 0:
        _run_git(["checkout", "--", rel])
    else:
        try:
            (REPO_PATH / rel).unlink(missing_ok=True)
        except OSError:
            pass


def _evolve_one_loss(loss: dict[str, Any]) -> dict[str, Any]:
    from .shared import ask_llm

    lesson = loss.get("lesson") or {}
    pointer_files = lesson.get("pointer_files") or []
    topic = lesson.get("topic") or loss.get("question", "")[:200]

    prompt = _EVOLVE_PROMPT.format(
        repo=str(REPO_PATH),
        topic=topic,
        pointer_files=json.dumps(pointer_files, ensure_ascii=False),
        question=loss.get("question", "")[:600],
        brain_answer=str(loss.get("brain_answer", ""))[:3000],
        no_brain_answer=str(loss.get("no_brain_answer", ""))[:3000],
        reason=str(loss.get("reason", ""))[:600],
    )
    text, backend, _meta = ask_llm(
        prompt,
        disable_brain=True,
        allowed_tools=["Read", "Grep", "Glob", "Edit"],
        cwd=str(REPO_PATH),
        timeout_s=EVOLVE_TIMEOUT_SECONDS,
    )
    payload = _extract_json(text) or {
        "kind": "skipped",
        "confidence": "low",
        "canonical_path": None,
        "pointers_added": [],
        "edits_applied": False,
        "summary": "agentic judge unavailable or unparseable",
        "escalation_reason": "judge_unavailable",
    }

    # Post-write guard: if the LLM edited a doc and the result contains
    # any brittle specific (line number, line range, file:line), revert
    # the edit and downgrade to skipped. This is the belt-and-suspenders
    # backstop on top of the prompt's hard rules.
    canonical = payload.get("canonical_path")
    if payload.get("edits_applied") and canonical:
        try:
            full = REPO_PATH / canonical
            if full.exists():
                content = full.read_text(encoding="utf-8", errors="replace")
                offender = _has_brittle_specifics(content)
                if offender:
                    _revert_brittle_edits(canonical)
                    payload = {
                        "kind": "skipped",
                        "confidence": "low",
                        "canonical_path": canonical,
                        "pointers_added": [],
                        "edits_applied": False,
                        "summary": f"reverted: brittle pattern {offender!r}",
                        "escalation_reason": "brittle_specifics_detected",
                    }
        except OSError:
            pass

    payload["question"] = loss.get("question", "")[:160]
    payload["backend"] = backend
    return payload


def _commit_all_edits(summary: str) -> dict[str, Any] | None:
    """One combined commit + push at the end of an evolve cycle. Same
    write lock brain_update + consistency use."""
    from .shared import _interprocess_write_lock, _run_git
    try:
        with _interprocess_write_lock():
            status = _run_git(["status", "--porcelain"])
            if not (status.stdout or "").strip():
                return None
            _run_git(["add", "-A"])
            msg = f"evolve: {summary[:80]}"
            commit = _run_git(["commit", "--quiet", "-m", msg])
            if commit.returncode != 0:
                return None
            _run_git(["push", "--quiet", "origin", "HEAD"])
            sha = _run_git(["rev-parse", "HEAD"]).stdout.strip()
            return {"sha": sha, "message": msg}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ── recheck pass: re-run A/B on the same losses, post-improvement ──────

def _build_evidence_block(question: str) -> str:
    """Build a fresh brain_evidence-style snippet bundle for the question.
    Skips the MCP layer; calls the vector index + Reads directly so we
    don't depend on a running MCP server."""
    try:
        from . import server as _server
        return _server.brain_evidence(question)
    except Exception:
        return ""


def _recheck_one(loss: dict[str, Any], evolve_id: str, cycle_id: str) -> dict[str, Any]:
    """Re-run A/B on one loss question. Tags the new eval row with
    trigger=f'evolve_recheck:{evolve_id}' so the brein.sh UI can group
    rows by evolution and show before/after."""
    from . import eval as _eval
    question = loss.get("question", "") or ""
    if not question:
        return {"question": "", "fired": False, "reason": "empty_question"}
    qhash = _eval._hash(question)
    trigger = f"evolve_recheck:{evolve_id}"
    evidence = _build_evidence_block(question)
    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "evolve_id": evolve_id,
        "event": "recheck_start", "question": question[:100], "trigger": trigger,
    })
    try:
        _eval._run_ab(question, evidence, trigger, qhash)
        fired = True
        err = None
    except Exception as exc:
        fired = False
        err = f"{type(exc).__name__}: {exc}"
    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "evolve_id": evolve_id,
        "event": "recheck_end", "question": question[:100], "trigger": trigger,
        "fired": fired, "error": err,
    })
    return {"question": question[:160], "fired": fired, "error": err}


def _run_rechecks(losses: list[dict[str, Any]], evolve_id: str, cycle_id: str) -> dict[str, Any]:
    """Fan rechecks across the same thread pool the improvement loop uses.
    Each recheck is one _run_ab call → one new row in eval-log.jsonl."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "evolve_id": evolve_id,
        "event": "recheck_cycle_start", "total": len(losses),
    })
    max_workers = max(1, min(EVOLVE_PARALLELISM, len(losses)))
    fired = errored = 0
    detail: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_recheck_one, l, evolve_id, cycle_id) for l in losses]
        for fut in as_completed(futs):
            r = fut.result()
            detail.append(r)
            if r.get("fired"):
                fired += 1
            else:
                errored += 1
    summary = {
        "evolve_id": evolve_id,
        "fired": fired,
        "errored": errored,
        "total": len(losses),
        "detail": detail,
    }
    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "evolve_id": evolve_id,
        "event": "recheck_cycle_end", "fired": fired, "errored": errored,
        "total": len(losses),
    })
    return summary


# ── log writer / reader ────────────────────────────────────────────────

def append_result(result: EvolveResult) -> None:
    EVOLVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVOLVE_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_json()) + "\n")


def _append_progress(row: dict[str, Any]) -> None:
    """Per-loss progress line. Safe to fail silently — progress logging
    is observability, not data."""
    try:
        EVOLVE_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVOLVE_PROGRESS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass


def read_log(limit: int = 20) -> list[dict[str, Any]]:
    if not EVOLVE_LOG_PATH.exists():
        return []
    lines = EVOLVE_LOG_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for l in lines[-limit:]:
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return out


# ── public entrypoints ─────────────────────────────────────────────────

def _evolve_one_loss_with_progress(
    i: int, total: int, loss: dict[str, Any], cycle_id: str,
    counters: dict[str, int], counters_lock: Any,
) -> dict[str, Any]:
    """Run one loss + write loss_start/loss_end progress rows around it.
    Designed for parallel execution — each call is independent except for
    the shared (lock-guarded) running counters."""
    import time as _time
    q_short = (loss.get("question", "") or "")[:100]
    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "event": "loss_start",
        "index": i, "total": total, "question": q_short,
    })
    t0 = _time.perf_counter()
    try:
        outcome = _evolve_one_loss(loss)
    except Exception as exc:
        outcome = {
            "kind": "error",
            "summary": f"{type(exc).__name__}: {exc}",
            "edits_applied": False,
            "question": loss.get("question", "")[:160],
        }
    elapsed = round(_time.perf_counter() - t0, 1)
    kind = outcome.get("kind")
    with counters_lock:
        if kind == "improved":
            counters["improved"] += 1
        elif kind == "escalated":
            counters["escalated"] += 1
        else:
            counters["skipped"] += 1
        snapshot = dict(counters)
    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "event": "loss_end",
        "index": i, "total": total,
        "question": q_short,
        "kind": kind,
        "summary": (outcome.get("summary") or "")[:240],
        "escalation_reason": outcome.get("escalation_reason"),
        "canonical_path": outcome.get("canonical_path"),
        "edits_applied": bool(outcome.get("edits_applied")),
        "elapsed_s": elapsed,
        "running_totals": snapshot,
    })
    return outcome


def run_evolve(limit: int = 50) -> EvolveResult:
    """Foreground evolve cycle. Reads recent no-brain wins and fans them
    out across EVOLVE_PARALLELISM threads (default 8). Each loss = one
    independent ask_llm call against (typically) a different canonical
    doc, so they parallelize cleanly. Combined commit + push at the end.

    Per-loss progress is logged to EVOLVE_PROGRESS_PATH including the
    agent's `summary` and `escalation_reason` so reasons are visible live.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    started = _now_iso()
    cycle_id = uuid.uuid4().hex[:8]
    losses = _read_recent_losses(limit=limit)
    total = len(losses)
    detail: list[dict[str, Any]] = []
    counters = {"improved": 0, "escalated": 0, "skipped": 0}
    counters_lock = threading.Lock()

    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "event": "cycle_start",
        "total_losses": total, "parallelism": EVOLVE_PARALLELISM,
    })

    if losses:
        max_workers = max(1, min(EVOLVE_PARALLELISM, total))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _evolve_one_loss_with_progress,
                    i, total, loss, cycle_id, counters, counters_lock,
                )
                for i, loss in enumerate(losses, start=1)
            ]
            for fut in as_completed(futures):
                detail.append(fut.result())

    improved = counters["improved"]
    escalated = counters["escalated"]
    skipped = counters["skipped"]

    evolve_id = uuid.uuid4().hex[:12]

    # Always attempt the commit — _commit_all_edits is a no-op when the repo
    # is clean. Earlier versions gated on `improved > 0` which silently lost
    # work when a parent run was killed mid-cycle (edits applied via the
    # Edit tool but never committed); the next cycle would correctly see
    # "docs already cover this", report skipped, and again not commit.
    commit_summary = f"{improved}/{len(losses)} losses patched (evolve_id={evolve_id})"
    commit_info = _commit_all_edits(commit_summary)
    sha = commit_info.get("sha") if isinstance(commit_info, dict) else None
    if commit_info:
        detail.append({"kind": "commit", **commit_info})

    # Recheck pass: re-run A/B on every loss question with trigger
    # 'evolve_recheck:<evolve_id>' so the eval log can show before/after
    # per evolution. Same parallelism cap as the improvement loop.
    if losses:
        recheck_summary = _run_rechecks(losses, evolve_id, cycle_id)
        detail.append({"kind": "recheck", **recheck_summary})

    result = EvolveResult(
        evolve_id=evolve_id,
        started_at=started,
        losses_examined=len(losses),
        losses_improved=improved,
        losses_escalated=escalated,
        losses_skipped=skipped,
        commit_sha=sha,
        losses=detail,
    )
    append_result(result)
    _append_progress({
        "ts": _now_iso(), "cycle_id": cycle_id, "event": "cycle_end",
        "total_losses": total,
        "improved": improved, "escalated": escalated, "skipped": skipped,
        "commit_sha": sha,
    })
    return result


def maybe_trigger_after_ab() -> int | None:
    """Called from eval._tick after a successful ab_run is logged.

    Fires a detached evolve worker every EVOLVE_TRIGGER_EVERY (default 50)
    ab_run rows. Recursion guard via EVOLVE_GUARD_ENV.
    """
    if os.environ.get(EVOLVE_GUARD_ENV) == "1":
        return None
    try:
        n = _count_ab_runs()
        if n <= 0 or n % EVOLVE_TRIGGER_EVERY != 0:
            return None
        return _spawn_detached()
    except Exception:
        return None


def _spawn_detached() -> int:
    log_path = EVOLVE_LOG_PATH.with_name("evolve-worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    cmd = [sys.executable, "-m", "brain_mcp.cli", "evolve", "run", "--quiet"]
    env = os.environ.copy()
    env[EVOLVE_GUARD_ENV] = "1"
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        env=env,
    )
    return proc.pid
