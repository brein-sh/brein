"""Continuous on-the-fly A/B eval.

After every brain_evidence call, decides whether this query is worth comparing.
If yes, spawns a background thread that:
  1. Re-asks the question with brain evidence → "with-brain" answer
  2. Re-asks the question with no evidence    → "no-brain" answer
  3. Judges which is better
  4. Appends one row to .brain/eval-log.jsonl

Inference uses **whatever client CLI is on PATH** — `claude`, `codex`, or
`gemini` — so it runs against your existing subscription auth. No separate
API key needed. Falls back to an OpenRouter env key if no CLI is found
(useful in CI / headless environments).

Triggers (cheap to detect, no extra inference):
  - dont_know  : answer text matched a "couldn't find / no record / not in"
                 pattern.
  - novel_hash : first time this query has been seen this process.

On by default. Failures are silently swallowed — eval must never break a
brain call. A recursion guard prevents a child claude/codex from
re-triggering this loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_PATH
from .shared import (
    LLM_GUARD_ENV as EVAL_GUARD_ENV,
    _which_llm_cli,
    ask_llm,
)

# ── Config ────────────────────────────────────────────────────────────────
EVAL_ENABLED = os.environ.get("BRAIN_EVAL_ENABLED", "on").lower() != "off"
EVAL_GUARDED = os.environ.get(EVAL_GUARD_ENV) == "1"

EVAL_LOG_PATH = REPO_PATH / ".brain" / "eval-log.jsonl"
EVAL_SEEN_PATH = Path(os.path.expanduser("~")) / ".brein" / "eval-seen.jsonl"
EVAL_DEDUP_HOURS = float(os.environ.get("BRAIN_EVAL_DEDUP_HOURS", "24"))

EVAL_CLI_TIMEOUT_S = float(os.environ.get("BRAIN_EVAL_CLI_TIMEOUT_S", "120"))

# OpenRouter env keys are read inside shared.ask_llm; the keepalive check
# below still needs to know if OpenRouter is configured.
_OR_KEY = os.environ.get("BRAIN_EVAL_OPENROUTER_KEY", "")

# ── Triggers ──────────────────────────────────────────────────────────────
_DONT_KNOW_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi (?:don't|do not) (?:have|know|see)\b",
        r"\bno (?:record|information|data|results?|mention|references?) (?:of|about|for|to)\b",
        r"\b(?:couldn't|can'?t|cannot|unable to) (?:find|locate|determine)\b",
        r"\bnot (?:in|covered by|documented in|present in|available in) the (?:repo|brain|docs|index)\b",
        r"\bnot enough (?:information|context|data)\b",
        r"\bisn'?t (?:in|covered|documented)\b",
    )
]

_seen_query_hashes: set[str] = set()
_seen_lock = threading.Lock()


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def _trigger(answer_text: str, query_hash: str) -> str | None:
    # In-process novel_hash is useless under stdio MCP (every tool call is a
    # fresh process) — persistent dedup happens in _tick instead. We just
    # surface whether the upstream text looked like a "don't know" so the
    # gate prompt can weight it higher.
    if not answer_text:
        return "novel"
    if any(p.search(answer_text) for p in _DONT_KNOW_PATTERNS):
        return "dont_know"
    return "novel"


# ── Persistent dedup + skipped-log helpers ──────────────────────────────

def _seen_recently(query_hash: str, hours: float = EVAL_DEDUP_HOURS) -> bool:
    """True if an eval (or gate decision) for this query_hash was recorded
    in the last `hours`. Cheap linear scan — file stays small."""
    if not EVAL_SEEN_PATH.exists():
        return False
    cutoff = time.time() - hours * 3600
    try:
        with EVAL_SEEN_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("query_hash") == query_hash and row.get("ts_unix", 0) >= cutoff:
                    return True
    except OSError:
        return False
    return False


def _mark_seen(query_hash: str, decision: str) -> None:
    try:
        EVAL_SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_SEEN_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "ts_unix": time.time(),
                "query_hash": query_hash,
                "decision": decision,
            }) + "\n")
    except OSError:
        pass


# ── Concurrent-fire dedup (O_EXCL claim slot) ──────────────────────────

# The 24h _seen_recently check is read-then-decide, with the LLM gate
# (~3-5s) sitting between read and write. Two workers spawned inside that
# window both see "not seen", both pass the gate, both run the full A/B.
# A claim slot — atomic O_EXCL create — closes the race: only the first
# worker to create the slot proceeds, the rest bail silently.
EVAL_CLAIMS_DIR = Path(os.path.expanduser("~")) / ".brein" / "eval-claims"
# Worker may crash before releasing. Treat older slot files as stale so
# the next eval for the same hash can claim again.
EVAL_CLAIM_STALE_SECONDS = float(os.environ.get("BRAIN_EVAL_CLAIM_STALE_SECONDS", "600"))


def _claim_path(query_hash: str) -> Path:
    # `:` is fine on macOS/linux but unusual; replace for portability.
    safe = query_hash.replace(":", "_").replace("/", "_")
    return EVAL_CLAIMS_DIR / f"{safe}.claim"


def _try_claim(query_hash: str) -> bool:
    """Return True if this worker should proceed, False if another worker
    has already claimed this query_hash. Uses O_EXCL for atomicity."""
    EVAL_CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    slot = _claim_path(query_hash)
    # Sweep a stale slot if present so a crashed worker doesn't block forever.
    try:
        age = time.time() - slot.stat().st_mtime
        if age > EVAL_CLAIM_STALE_SECONDS:
            slot.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError:
        pass
    try:
        fd = os.open(str(slot), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    except OSError:
        return False
    try:
        os.write(fd, f"{os.getpid()} {time.time()}\n".encode())
    finally:
        os.close(fd)
    return True


def _release_claim(query_hash: str) -> None:
    try:
        _claim_path(query_hash).unlink(missing_ok=True)
    except OSError:
        pass


def _log_gate_skip(question: str, query_hash: str, trigger: str, gate_meta: dict) -> None:
    """Write a skipped-row to eval-log.jsonl so we can see the gate worked
    without spending the full A/B budget."""
    try:
        EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": "gate_skipped",
                "query_hash": query_hash,
                "trigger": trigger,
                "question": question[:500],
                "gate": gate_meta,
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _commit_and_push_eval_row(commit_msg: str) -> None:
    """Commit the just-appended eval row and push. Silent on every failure
    — telemetry must never break the host. Uses the same inter-process lock
    as brain_update so we don't race writes.

    Skipped when EVAL_LOG_PATH lives outside the brain git repo (e.g. user
    points BRAIN_RETRIEVAL_LOG/eval at a path under $HOME for purely-local
    telemetry)."""
    try:
        # eval log lives at REPO_PATH/.brain/eval-log.jsonl by construction.
        # Bail if someone moved it outside the repo.
        try:
            rel = EVAL_LOG_PATH.relative_to(REPO_PATH)
        except ValueError:
            return
        from .shared import _interprocess_write_lock, _run_git
        with _interprocess_write_lock():
            _run_git(["pull", "--ff-only", "--quiet", "origin", "HEAD"])
            r = _run_git(["add", str(rel)])
            if r.returncode != 0:
                return
            # Nothing staged? (e.g. log was reverted between append+commit) → skip
            staged = _run_git(["diff", "--cached", "--quiet"])
            if staged.returncode == 0:
                return
            c = _run_git(["commit", "-q", "-m", commit_msg])
            if c.returncode != 0:
                return
            _run_git(["push", "--quiet", "origin", "HEAD"])
    except Exception:
        pass


# ── Inference: thin shims around shared.ask_llm ──────────────────────────
def _which_cli() -> str | None:
    return _which_llm_cli()


def _ask(prompt: str, disable_brain: bool = False) -> tuple[str, str, dict[str, Any]]:
    """One-shot ask, no tools. Used by gate and per-arm success classifier."""
    return ask_llm(prompt, disable_brain=disable_brain, timeout_s=EVAL_CLI_TIMEOUT_S)


# Tools the A/B arms are allowed to use. Read/Grep/Glob = navigation;
# Bash = git, ls, anything else. Edit deliberately omitted — the eval
# is read-only; nothing should mutate the user's repos.
_AB_TOOLS = ["Read", "Grep", "Glob", "Bash"]


def _ask_agentic(prompt: str, disable_brain: bool = False) -> tuple[str, str, dict[str, Any]]:
    """Agentic ask with real tools. Used for the A/B arms so we capture
    a trajectory (which files were read, what was grepped)."""
    return ask_llm(
        prompt,
        disable_brain=disable_brain,
        allowed_tools=_AB_TOOLS,
        timeout_s=EVAL_CLI_TIMEOUT_S,
    )


# ── Per-arm success classifier (replaces pairwise "who's better") ───────
#
# Why this shape: the old pairwise judge rewarded "more specific" and so
# punished the brain on any "where is X" question (the no-brain agent's
# fresh grep always cites concrete paths). The evolve loop then learned to
# paste those paths into memory, which immediately rot. The new design
# scores each arm INDEPENDENTLY on a binary "did it answer the question?"
# and lets the trajectory diff (in _make_lesson) decide what's worth
# learning. One LLM call, no comparison surface for shortcut bias.

def _classify_success(question: str, brain_answer: str, no_brain_answer: str) -> tuple[bool, bool, dict[str, Any]]:
    prompt = (
        "Two assistants tried to answer the same question. For each, decide "
        "whether the answer actually addresses the question with a real, "
        "useful answer (true) or whether it failed / said it couldn't find "
        "anything / gave a vague non-answer (false). Judge each independently.\n\n"
        f"Question: {question[:1000]}\n\n"
        f"Answer A:\n{brain_answer[:2000]}\n\n"
        f"Answer B:\n{no_brain_answer[:2000]}\n\n"
        'Respond with JSON only: {"a_success": true|false, "b_success": true|false}'
    )
    raw, _, meta = _ask(prompt, disable_brain=True)
    if not raw:
        return False, False, meta
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
            return bool(parsed.get("a_success", False)), bool(parsed.get("b_success", False)), meta
    except Exception:
        pass
    return False, False, meta


# Files under these dirs aren't "load-bearing pointers" — they're noise
# the brain shouldn't learn to point at.
_LESSON_PATH_EXCLUDES = (
    str(REPO_PATH),                                       # the brain repo itself
    os.path.expanduser("~/.brein"),                       # brain runtime state
    "/tmp/",
    "/private/tmp/",
)


def _module_of(path: str) -> str:
    """Module-granularity bucket for a file path. Strip filename → directory.
    Used to dedup `foo/bar/x.py` and `foo/bar/y.py` to `foo/bar`."""
    p = path.rstrip("/")
    if "/" in p:
        return p.rsplit("/", 1)[0]
    return p


def _diff_trajectories(brain_traj: dict[str, Any], no_brain_traj: dict[str, Any]) -> list[str]:
    """Files the no-brain arm read that the brain arm never opened, at
    module granularity. These are the candidate pointers the brain is
    missing. Returns ordered, deduped list — empty if no diff."""
    brain_modules = {_module_of(f) for f in (brain_traj.get("files_read") or [])}
    pointers: list[str] = []
    seen: set[str] = set()
    for f in (no_brain_traj.get("files_read") or []):
        if any(f.startswith(prefix) for prefix in _LESSON_PATH_EXCLUDES):
            continue
        mod = _module_of(f)
        if mod in brain_modules or mod in seen:
            continue
        seen.add(mod)
        pointers.append(f)
    return pointers


def _make_lesson(
    question: str,
    brain_success: bool, no_brain_success: bool,
    brain_traj: dict[str, Any], no_brain_traj: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a lesson the evolve loop should learn from, or None.

    Strong lesson  : brain failed AND no_brain succeeded → the brain misrouted
                     or stayed silent; the files no_brain found are the missing
                     pointers.
    Soft lesson    : both succeeded BUT no_brain used materially fewer tool
                     calls AND uncovered files brain never touched → brain's
                     pointer set is incomplete.
    Otherwise      : None. Nothing to learn (brain won, both failed, or the
                     no-brain win was just better prose with no new files).
    """
    pointers = _diff_trajectories(brain_traj, no_brain_traj)
    if not pointers:
        return None
    if not brain_success and no_brain_success:
        return {"reason": "brain_failed_no_brain_found", "pointer_files": pointers[:8], "topic": question[:200]}
    brain_calls = int(brain_traj.get("num_tool_calls") or 0)
    no_brain_calls = int(no_brain_traj.get("num_tool_calls") or 0)
    if (brain_success and no_brain_success
            and no_brain_calls > 0
            and brain_calls >= no_brain_calls + 3):
        return {"reason": "brain_inefficient_missing_pointer", "pointer_files": pointers[:8], "topic": question[:200]}
    return None


def _derive_verdict(brain_success: bool, no_brain_success: bool, lesson: dict[str, Any] | None) -> str:
    """Backward-compat label for dashboards. Not used by the loop."""
    if brain_success and not no_brain_success:
        return "brain_better"
    if no_brain_success and not brain_success:
        return "no_brain_better"
    if lesson is not None:
        return "no_brain_better"  # had something to teach → de facto loss
    return "tie"


def _run_ab(question: str, evidence_block: str, trigger: str, query_hash: str) -> None:
    """Background worker. Generates both answers, judges, appends row.

    Both arms run as full CLI sessions with all default tools (filesystem,
    grep, web, etc.) — symmetric with the manual brain-vs-repo eval. The
    *only* difference is whether the brain MCP is available:
      - brain arm: brain MCP + evidence already injected into the prompt
      - no-brain arm: BRAIN_EVAL_DISABLE=1 → brain MCP refuses to start
    """
    brain_prompt = (
        "Answer the following question. You have brain knowledge-base evidence "
        "below, AND you are running in the user's home directory with full "
        "filesystem and web tools — you can grep, read, and search any "
        "repos or files you find under here to verify or supplement the "
        "evidence. If the answer isn't supported, say so plainly.\n\n"
        f"Question: {question}\n\nEvidence from brain:\n{evidence_block[:8000]}"
    )
    no_brain_prompt = (
        "Answer the following question. You are running in the user's home "
        "directory with full filesystem and web tools — actively explore "
        "(ls, grep, read) any repos or files you find under here that look "
        "relevant. Don't give up after one search; navigate into likely "
        "subdirectories. If after a real search you still can't find it, "
        "say so plainly — no hallucinated specifics.\n\n"
        f"Question: {question}"
    )

    brain_answer, backend, brain_meta = _ask_agentic(brain_prompt, disable_brain=False)
    no_brain, _, no_brain_meta = _ask_agentic(no_brain_prompt, disable_brain=True)

    if not brain_answer and not no_brain:
        return

    brain_success, no_brain_success, judge_meta = _classify_success(
        question, brain_answer, no_brain,
    )
    brain_traj = brain_meta.get("trajectory") or {}
    no_brain_traj = no_brain_meta.get("trajectory") or {}
    lesson = _make_lesson(question, brain_success, no_brain_success, brain_traj, no_brain_traj)
    verdict = _derive_verdict(brain_success, no_brain_success, lesson)
    reason = (lesson or {}).get("reason", "no_lesson")
    # Kept for dashboard back-compat; no longer drives the loop.
    question_class = "unknown"
    brain_admitted_no_answer = bool(brain_answer) and not brain_success

    def _num(d: dict[str, Any], k: str) -> float:
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else 0.0

    total_cost_usd = _num(brain_meta, "cost_usd") + _num(no_brain_meta, "cost_usd") + _num(judge_meta, "cost_usd")
    total_input_tokens = int(_num(brain_meta, "input_tokens") + _num(no_brain_meta, "input_tokens") + _num(judge_meta, "input_tokens"))
    total_output_tokens = int(_num(brain_meta, "output_tokens") + _num(no_brain_meta, "output_tokens") + _num(judge_meta, "output_tokens"))
    total_cache_read_tokens = int(_num(brain_meta, "cache_read_input_tokens") + _num(no_brain_meta, "cache_read_input_tokens") + _num(judge_meta, "cache_read_input_tokens"))
    total_wall_clock_ms = _num(brain_meta, "latency_ms") + _num(no_brain_meta, "latency_ms") + _num(judge_meta, "latency_ms")

    row: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_hash": query_hash,
        "trigger": trigger,
        "question": question[:1000],
        "brain_answer": brain_answer[:2000],
        "no_brain_answer": no_brain[:2000],
        "verdict": verdict,
        "reason": reason,
        "question_class": question_class,
        "brain_admitted_no_answer": brain_admitted_no_answer,
        "brain_success": brain_success,
        "no_brain_success": no_brain_success,
        "lesson": lesson,
        "backend": backend,
        # Per-arm metadata — wall-clock, tokens, cost, turns, cache hits, etc.
        "brain_meta": brain_meta,
        "no_brain_meta": no_brain_meta,
        "judge_meta": judge_meta,
        # Roll-ups for easy aggregation (no nested digging needed).
        "totals": {
            "cost_usd": round(total_cost_usd, 6),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_input_tokens": total_cache_read_tokens,
            "wall_clock_ms": round(total_wall_clock_ms, 1),
        },
    }

    try:
        EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _llm_gate(question: str, evidence_block: str) -> tuple[bool, dict[str, Any]]:
    """Cheap LLM judge: is this query worth the full A/B benchmark?

    One short prompt, one-word answer. Runs without the brain MCP so the
    gate decision can't be biased by the very system we're evaluating.
    """
    prompt = (
        "You gate a costly background A/B benchmark on the user's knowledge "
        "base. Run it ONLY when the query is significant: a turning point in "
        "a conversation, an essential fact-check, a high-stakes decision, or "
        "a question whose answer would change real action. SKIP routine "
        "browsing, trivial lookups, name pings, repeated checks.\n\n"
        f"Question: {question[:500]}\n"
        f"Brain evidence preview: {evidence_block[:500]}\n\n"
        "Answer with one word only: yes or no."
    )
    text, _backend, meta = _ask(prompt, disable_brain=True)
    answer = (text or "").strip().lower()
    decided = answer.startswith("y")
    return decided, {"raw": (text or "")[:80], "decision": "yes" if decided else "no", **meta}


def _tick(question: str, evidence_block: str, query_hash: str, trigger: str) -> None:
    """Detached worker: dedup → LLM gate → conditional A/B → log.

    Always called in a detached subprocess so it survives the MCP server
    process exit. Failures are silently swallowed.
    """
    try:
        if _seen_recently(query_hash):
            return
        if not _which_cli() and not _OR_KEY:
            return
        # Atomic claim — second concurrent worker for the same hash bails
        # here, even if it slipped past _seen_recently (the gate window
        # gave it a chance to race).
        if not _try_claim(query_hash):
            return
        try:
            decided, gate_meta = _llm_gate(question, evidence_block)
            if not decided:
                _mark_seen(query_hash, "gate_skipped")
                _log_gate_skip(question, query_hash, trigger, gate_meta)
                _commit_and_push_eval_row(
                    f"eval(gate_skipped): {question[:60].splitlines()[0]}"
                )
                return
            _mark_seen(query_hash, "ab_run")
            _run_ab(question, evidence_block, trigger, query_hash)
            _commit_and_push_eval_row(
                f"eval(ab): {question[:60].splitlines()[0]}"
            )
            # Self-improvement: every N ab_runs, spawn an agentic worker
            # that reads recent no-brain wins and amends the canonical
            # brain doc with verified file paths / line refs. Detached;
            # no impact on this eval row's commit.
            try:
                from . import evolve as _evolve
                _evolve.maybe_trigger_after_ab()
            except Exception:
                pass
        finally:
            _release_claim(query_hash)
    except Exception:
        # Telemetry must never break the host.
        pass


def maybe_eval(question: str, evidence_block: str, brain_answer_preview: str = "") -> None:
    """Public entrypoint. Spawns a detached subprocess so the eval survives
    the MCP server process exiting (which it does immediately after every
    stdio tool call). Non-blocking from the caller's perspective."""
    if not EVAL_ENABLED or EVAL_GUARDED:
        return
    if not question.strip():
        return
    if not _which_cli() and not _OR_KEY:
        return
    query_hash = _hash(question)
    trigger = _trigger(brain_answer_preview or evidence_block, query_hash) or "novel"
    # Cheap pre-spawn dedup — saves a fork if we already know we'd skip.
    if _seen_recently(query_hash):
        return
    _spawn_eval_worker(question, evidence_block, query_hash, trigger)


def _spawn_eval_worker(question: str, evidence_block: str, query_hash: str, trigger: str) -> None:
    """Double-fork-equivalent: start_new_session detaches us from the parent
    process group so the MCP server's exit doesn't take us down."""
    payload = json.dumps({
        "question": question,
        "evidence_block": evidence_block,
        "query_hash": query_hash,
        "trigger": trigger,
    })
    log_path = EVAL_LOG_PATH.parent / "eval-worker.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab", buffering=0)
    except OSError:
        log_fh = subprocess.DEVNULL  # type: ignore[assignment]
    import sys as _sys  # local import to avoid touching module-level deps
    # Use the SAME Python interpreter the server is running under, via the
    # importable module. `shutil.which("brein")` would resolve to a
    # globally-installed copy that may be older than the running code (the
    # `eval` subcommand might not exist there yet). The module path always
    # matches the running process.
    cmd = [_sys.executable, "-m", "brain_mcp.cli", "eval", "tick"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(payload.encode("utf-8"))
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    except (FileNotFoundError, OSError):
        # No brein on PATH (uncommon) — silently degrade. The synchronous
        # `brain-eval` CLI still works.
        pass
