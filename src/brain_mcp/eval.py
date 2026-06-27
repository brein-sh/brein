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
import shutil
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_PATH

# ── Config ────────────────────────────────────────────────────────────────
EVAL_ENABLED = os.environ.get("BRAIN_EVAL_ENABLED", "on").lower() != "off"

# Recursion guard: when we subprocess claude/codex, we set this env var so
# the child's brain_evidence call doesn't trigger another eval.
EVAL_GUARD_ENV = "BRAIN_EVAL_IN_PROGRESS"
EVAL_GUARDED = os.environ.get(EVAL_GUARD_ENV) == "1"

EVAL_LOG_PATH = REPO_PATH / ".brain" / "eval-log.jsonl"
EVAL_SEEN_PATH = Path(os.path.expanduser("~")) / ".brein" / "eval-seen.jsonl"
EVAL_DEDUP_HOURS = float(os.environ.get("BRAIN_EVAL_DEDUP_HOURS", "24"))

# CLI preference order; first match wins. Override with BRAIN_EVAL_CLIENT.
_CLI_PREFERENCE = (os.environ.get("BRAIN_EVAL_CLIENT") or "claude,codex,gemini").split(",")
EVAL_CLI_TIMEOUT_S = float(os.environ.get("BRAIN_EVAL_CLI_TIMEOUT_S", "120"))

# OpenRouter fallback (only used if no CLI is on PATH).
_OR_KEY = os.environ.get("BRAIN_EVAL_OPENROUTER_KEY", "")
_OR_MODEL = os.environ.get("BRAIN_EVAL_MODEL", "deepseek/deepseek-v4-flash")
_OR_TIMEOUT_S = float(os.environ.get("BRAIN_EVAL_TIMEOUT_S", "60"))

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


# ── Inference: CLI first, OpenRouter fallback ────────────────────────────
def _which_cli() -> str | None:
    """Pick the first available CLI from preference order."""
    for name in _CLI_PREFERENCE:
        name = name.strip()
        if name and shutil.which(name):
            return name
    return None


def _cli_completion(cli: str, prompt: str, disable_brain: bool = False) -> tuple[str, dict[str, Any]]:
    """Run a headless completion via `<cli> -p <prompt>` in a subprocess.

    Returns (text, meta). text is "" on any failure. meta is a dict capturing
    everything we can observe about the call: wall-clock latency, stdout/stderr
    sizes, returncode, timeout flag, prompt size, and — for claude — the rich
    JSON envelope (input/output tokens, cache tokens, cost USD, num_turns,
    api duration). For codex/gemini we just record the shell-level stats.
    """
    env = os.environ.copy()
    env[EVAL_GUARD_ENV] = "1"
    if disable_brain:
        env["BRAIN_EVAL_DISABLE"] = "1"
    cwd = os.path.expanduser("~")
    prompt_chars = len(prompt)

    # Prefer claude's structured JSON output — gives us tokens + cost + turns.
    use_json = (cli == "claude")
    cmd = [cli, "-p", prompt] + (["--output-format", "json"] if use_json else [])

    meta: dict[str, Any] = {
        "cli": cli,
        "disable_brain": disable_brain,
        "prompt_chars": prompt_chars,
        "prompt_tokens_est": prompt_chars // 4,
        "timed_out": False,
    }
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=EVAL_CLI_TIMEOUT_S,
            env=env,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        meta["timed_out"] = True
        meta["returncode"] = None
        return "", meta
    except Exception as exc:
        meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return "", meta

    meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    meta["returncode"] = r.returncode
    meta["stdout_chars"] = len(r.stdout or "")
    meta["stderr_chars"] = len(r.stderr or "")

    if r.returncode != 0:
        return "", meta

    raw = (r.stdout or "").strip()
    text = raw

    # Parse claude's JSON envelope when present.
    if use_json and raw.startswith("{"):
        try:
            payload = json.loads(raw)
            text = str(payload.get("result", "") or "").strip()
            usage = payload.get("usage", {}) or {}
            meta["input_tokens"] = usage.get("input_tokens")
            meta["output_tokens"] = usage.get("output_tokens")
            meta["cache_creation_input_tokens"] = usage.get("cache_creation_input_tokens")
            meta["cache_read_input_tokens"] = usage.get("cache_read_input_tokens")
            meta["cost_usd"] = payload.get("total_cost_usd")
            meta["num_turns"] = payload.get("num_turns")
            meta["duration_ms"] = payload.get("duration_ms")
            meta["duration_api_ms"] = payload.get("duration_api_ms")
            meta["session_id"] = payload.get("session_id")
            meta["is_error"] = payload.get("is_error", False)
        except Exception:
            # Fall back to raw text if envelope parse fails.
            meta["json_parse_failed"] = True

    meta["answer_chars"] = len(text)
    meta["answer_tokens_est"] = len(text) // 4
    return text, meta


def _openrouter_completion(prompt: str) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {
        "cli": None,
        "openrouter_model": _OR_MODEL,
        "prompt_chars": len(prompt),
        "prompt_tokens_est": len(prompt) // 4,
    }
    if not _OR_KEY:
        meta["error"] = "no_openrouter_key"
        return "", meta
    body = {
        "model": _OR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_OR_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://brein.sh",
            "X-Title": "brain-mcp-eval",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=_OR_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = str(data["choices"][0]["message"]["content"])
            meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            usage = data.get("usage", {}) or {}
            meta["input_tokens"] = usage.get("prompt_tokens")
            meta["output_tokens"] = usage.get("completion_tokens")
            meta["total_tokens"] = usage.get("total_tokens")
            meta["answer_chars"] = len(text)
            meta["answer_tokens_est"] = len(text) // 4
            return text, meta
    except Exception as exc:
        meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return "", meta


def _ask(prompt: str, disable_brain: bool = False) -> tuple[str, str, dict[str, Any]]:
    """Run a prompt through the best available backend. Returns (text, backend, meta).
    disable_brain=True is only meaningful for the CLI backend (sets
    BRAIN_EVAL_DISABLE=1 on the child process so brain MCP refuses to start).
    """
    cli = _which_cli()
    if cli:
        tag = f"cli:{cli}+tools" + (":no-brain" if disable_brain else "")
        text, meta = _cli_completion(cli, prompt, disable_brain=disable_brain)
        return text, tag, meta
    text, meta = _openrouter_completion(prompt)
    return text, f"openrouter:{_OR_MODEL}", meta


# ── Judge ────────────────────────────────────────────────────────────────
def _judge(question: str, brain_answer: str, no_brain_answer: str) -> tuple[str, str, str, dict[str, Any]]:
    prompt = (
        "Compare two AI assistants answering the same question. Pick which is "
        "more helpful, accurate, and specific. If they are equivalent or both "
        "fail, say tie.\n\n"
        "Also classify the question:\n"
        "  - internal_only: only answerable with company-specific knowledge "
        "(internal facts, decisions, people, dates). General training data alone cannot answer.\n"
        "  - general: answerable from general training-data knowledge alone, no company context needed.\n"
        "  - mixed: needs both — a general topic with a company-specific angle.\n\n"
        f"Question: {question[:1000]}\n\n"
        f"Answer A (with brain retrieval):\n{brain_answer[:2000]}\n\n"
        f"Answer B (no brain, model knowledge only):\n{no_brain_answer[:2000]}\n\n"
        'Respond with JSON only: {"verdict": "brain_better" | "no_brain_better" | "tie", '
        '"reason": "one short sentence", '
        '"question_class": "internal_only" | "general" | "mixed"}'
    )
    raw, _, meta = _ask(prompt, disable_brain=True)
    if not raw:
        return "tie", "judge_unavailable", "unknown", meta
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
            verdict = parsed.get("verdict", "tie")
            if verdict not in {"brain_better", "no_brain_better", "tie"}:
                verdict = "tie"
            qclass = parsed.get("question_class", "unknown")
            if qclass not in {"internal_only", "general", "mixed"}:
                qclass = "unknown"
            return verdict, str(parsed.get("reason", ""))[:240], qclass, meta
    except Exception:
        pass
    return "tie", "judge_parse_error", "unknown", meta


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

    brain_answer, backend, brain_meta = _ask(brain_prompt, disable_brain=False)
    no_brain, _, no_brain_meta = _ask(no_brain_prompt, disable_brain=True)

    if not brain_answer and not no_brain:
        return

    verdict, reason, question_class, judge_meta = _judge(question, brain_answer, no_brain)

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
