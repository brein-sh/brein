"""Stop-hook backend: if the agent just answered something and didn't
write anything to the brain this turn, nudge it to brain_update before
stopping. No clever entity analysis — just "write often."

The model decides whether the turn actually held new knowledge. Honors
stop_hook_active so it can't loop forever."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_MIN_ANSWER_CHARS = 100  # skip "ok"/"yes"/trivial replies, nothing else


def _write_flag_path() -> Path:
    session = os.environ.get("CLAUDE_CODE_SESSION_ID", "default")
    return Path(f"/tmp/claude-brein-write-{session}")


def _last_assistant_char_count(transcript_path: str) -> int:
    """Total text-block chars in the trailing run of assistant messages.
    Used only to skip trivial replies; we don't inspect the content."""
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return 0
    msgs = []
    for line in lines:
        try:
            msgs.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    total = 0
    for m in reversed(msgs):
        msg_type = m.get("type")
        role = (m.get("message") or {}).get("role") if isinstance(m.get("message"), dict) else None
        kind = role or msg_type
        if kind == "user":
            if total:
                break
            continue
        if kind != "assistant":
            continue
        content = (m.get("message") or {}).get("content") or m.get("content") or ""
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    total += len(p.get("text", ""))
        else:
            total += len(str(content))
    return total


def run() -> int:
    try:
        env = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return 0

    if env.get("stop_hook_active"):
        return 0
    if _write_flag_path().exists():
        return 0

    transcript_path = env.get("transcript_path") or ""
    if not transcript_path or not Path(transcript_path).is_file():
        return 0
    if _last_assistant_char_count(transcript_path) < _MIN_ANSWER_CHARS:
        return 0

    reason = (
        "Did you learn anything this turn that isn't in the brain yet? "
        "If yes — names, decisions, code paths, anything durable — call "
        "brain_update now. If it's truly already covered or trivial, just "
        "stop again."
    )
    sys.stdout.write(json.dumps({"decision": "block", "reason": reason}) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via CLI
    sys.exit(run())
