"""Continuous on-the-fly A/B eval.

After every brain_answer call, decides whether this query is worth comparing.
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_PATH

# ── Config ────────────────────────────────────────────────────────────────
EVAL_ENABLED = os.environ.get("BRAIN_EVAL_ENABLED", "on").lower() != "off"

# Recursion guard: when we subprocess claude/codex, we set this env var so
# the child's brain_answer call doesn't trigger another eval.
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


def _cli_completion(cli: str, prompt: str) -> str:
    """Run a headless completion via `<cli> -p <prompt>` in a subprocess.
    Returns "" on any failure. Sets a recursion-guard env var so a brain
    call inside the child doesn't loop back into us.
    """
    env = os.environ.copy()
    env[EVAL_GUARD_ENV] = "1"
    try:
        r = subprocess.run(
            [cli, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=EVAL_CLI_TIMEOUT_S,
            env=env,
        )
        if r.returncode != 0:
            return ""
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _openrouter_completion(prompt: str) -> str:
    if not _OR_KEY:
        return ""
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
            "HTTP-Referer": "https://brainincorp.com",
            "X-Title": "brain-mcp-eval",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_OR_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _ask(prompt: str) -> tuple[str, str]:
    """Run a prompt through the best available backend. Returns (text, backend)."""
    cli = _which_cli()
    if cli:
        return _cli_completion(cli, prompt), f"cli:{cli}"
    return _openrouter_completion(prompt), f"openrouter:{_OR_MODEL}"


# ── Judge ────────────────────────────────────────────────────────────────
def _judge(question: str, brain_answer: str, no_brain_answer: str) -> tuple[str, str]:
    prompt = (
        "Compare two AI assistants answering the same question. Pick which is "
        "more helpful, accurate, and specific. If they are equivalent or both "
        "fail, say tie.\n\n"
        f"Question: {question[:1000]}\n\n"
        f"Answer A (with brain retrieval):\n{brain_answer[:2000]}\n\n"
        f"Answer B (no brain, model knowledge only):\n{no_brain_answer[:2000]}\n\n"
        'Respond with JSON only: {"verdict": "brain_better" | "no_brain_better" | "tie", '
        '"reason": "one short sentence"}'
    )
    raw, _ = _ask(prompt)
    if not raw:
        return "tie", "judge_unavailable"
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
            verdict = parsed.get("verdict", "tie")
            if verdict not in {"brain_better", "no_brain_better", "tie"}:
                verdict = "tie"
            return verdict, str(parsed.get("reason", ""))[:240]
    except Exception:
        pass
    return "tie", "judge_parse_error"


def _run_ab(question: str, evidence_block: str, trigger: str, query_hash: str) -> None:
    """Background worker. Generates both answers, judges, appends row."""
    brain_prompt = (
        "You are a knowledge-base assistant. Answer the question using ONLY "
        "the evidence below. If the evidence doesn't cover it, say so plainly.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence_block[:8000]}"
    )
    no_brain_prompt = (
        "Answer the question from general knowledge only. If you don't know, "
        "say so plainly. No hallucinated specifics.\n\n"
        f"Question: {question}"
    )

    brain_answer, backend = _ask(brain_prompt)
    no_brain, _ = _ask(no_brain_prompt)

    if not brain_answer and not no_brain:
        return

    verdict, reason = _judge(question, brain_answer, no_brain)

    row: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_hash": query_hash,
        "trigger": trigger,
        "question": question[:1000],
        "brain_answer": brain_answer[:2000],
        "no_brain_answer": no_brain[:2000],
        "verdict": verdict,
        "reason": reason,
        "backend": backend,
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
