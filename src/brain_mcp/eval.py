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
    if not answer_text:
        return None
    if any(p.search(answer_text) for p in _DONT_KNOW_PATTERNS):
        return "dont_know"
    with _seen_lock:
        if query_hash not in _seen_query_hashes:
            _seen_query_hashes.add(query_hash)
            return "novel_hash"
    return None


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


def maybe_eval(question: str, evidence_block: str, brain_answer_preview: str = "") -> None:
    """Public entrypoint. Non-blocking. Fire-and-forget daemon thread."""
    if not EVAL_ENABLED or EVAL_GUARDED:
        return
    if not question.strip():
        return
    # Need *some* inference backend.
    if not _which_cli() and not _OR_KEY:
        return
    query_hash = _hash(question)
    trigger = _trigger(brain_answer_preview or evidence_block, query_hash)
    if not trigger:
        return
    threading.Thread(
        target=_run_ab,
        args=(question, evidence_block, trigger, query_hash),
        name="brain-eval-ab",
        daemon=True,
    ).start()
