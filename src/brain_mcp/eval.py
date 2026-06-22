"""Continuous on-the-fly A/B eval.

After every brain_answer call, decides whether this query is worth comparing.
If yes, spawns a background thread that:
  1. Re-asks the question with brain evidence → "with-brain" answer
  2. Re-asks the question with no evidence    → "no-brain" answer
  3. Judges which is better with a cheap model
  4. Appends one row to .brain/eval-log.jsonl

Triggers (cheap to detect, no extra inference):
  - dont_know  : answer text matched a "don't know" pattern
  - novel_hash : first time this query has been seen this process

Everything is env-gated. Off by default. Off if no OpenRouter key.
Failures are swallowed — eval must never break a brain call.

Stdlib only (urllib.request). No new dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .config import REPO_PATH

EVAL_ENABLED = os.environ.get("BRAIN_EVAL_ENABLED", "off").lower() == "on"
EVAL_LOG_PATH = REPO_PATH / ".brain" / "eval-log.jsonl"
EVAL_OPENROUTER_KEY = os.environ.get("BRAIN_EVAL_OPENROUTER_KEY", "")
EVAL_MODEL = os.environ.get("BRAIN_EVAL_MODEL", "deepseek/deepseek-v4-flash")
EVAL_JUDGE_MODEL = os.environ.get("BRAIN_EVAL_JUDGE_MODEL", "deepseek/deepseek-v4-flash")
EVAL_REQUEST_TIMEOUT = float(os.environ.get("BRAIN_EVAL_TIMEOUT_S", "60"))

# ponytail: regex is good enough for v1. v2 is the model self-tagging its
# own confidence in the final response. List is small + extensible.
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

# Ephemeral in-process set — restarts on every MCP boot. Fine for "first
# occurrence per process" semantics; the dataset already grows over time.
_seen_query_hashes: set[str] = set()
_seen_lock = threading.Lock()


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def _trigger(answer_text: str, query_hash: str) -> str | None:
    """Decide if this query is worth A/B-testing. Returns reason or None."""
    if not answer_text:
        return None
    if any(p.search(answer_text) for p in _DONT_KNOW_PATTERNS):
        return "dont_know"
    with _seen_lock:
        if query_hash not in _seen_query_hashes:
            _seen_query_hashes.add(query_hash)
            return "novel_hash"
    return None


def _openrouter_chat(model: str, messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
    """Sync OpenRouter call via stdlib. Returns "" on any failure."""
    if not EVAL_OPENROUTER_KEY:
        return ""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {EVAL_OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://brainincorp.com",
            "X-Title": "brain-mcp-eval",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EVAL_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _judge(question: str, brain_answer: str, no_brain_answer: str) -> tuple[str, str]:
    """Cheap LLM judge. Returns (verdict, reason)."""
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
    raw = _openrouter_chat(EVAL_JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=200)
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


def _run_ab(question: str, evidence_block: str, brain_answer: str, trigger: str, query_hash: str) -> None:
    """Background worker. Generates no_brain answer, judges, appends row."""
    if not EVAL_OPENROUTER_KEY:
        return

    # If the caller didn't pre-synthesize a brain answer, do it now using the
    # evidence block. Keeps the comparison apples-to-apples.
    if not brain_answer:
        brain_answer = _openrouter_chat(
            EVAL_MODEL,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge-base assistant. Answer the question using ONLY "
                        "the evidence below. If the evidence doesn't cover it, say so plainly."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{evidence_block[:8000]}"},
            ],
            max_tokens=512,
        )

    no_brain = _openrouter_chat(
        EVAL_MODEL,
        [
            {
                "role": "system",
                "content": (
                    "Answer the question from general knowledge only. If you don't know, "
                    "say so plainly. No hallucinated specifics."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=512,
    )

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
        "eval_model": EVAL_MODEL,
        "judge_model": EVAL_JUDGE_MODEL,
    }

    try:
        EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # ponytail: eval write must never break anything upstream.
        pass


def maybe_eval(question: str, evidence_block: str, brain_answer_preview: str = "") -> None:
    """Public entrypoint. Called from brain_answer after it returns to the
    client. Non-blocking — kicks off a daemon thread and returns immediately.

    `brain_answer_preview` is the synthesized response we want to compare. If
    empty, the background worker synthesizes one using the evidence.
    """
    if not EVAL_ENABLED or not EVAL_OPENROUTER_KEY:
        return
    if not question.strip():
        return
    query_hash = _hash(question)
    trigger = _trigger(brain_answer_preview or evidence_block, query_hash)
    if not trigger:
        return
    threading.Thread(
        target=_run_ab,
        args=(question, evidence_block, brain_answer_preview, trigger, query_hash),
        name="brain-eval-ab",
        daemon=True,
    ).start()
