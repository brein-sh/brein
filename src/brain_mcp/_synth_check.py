"""Stop-hook backend: when the agent finishes an answer and hasn't written
to the brain, fan out a detached side-agent that does the brain_update.
The user-visible Stop is never blocked — control returns immediately and
the side-agent runs in the background.

The side-agent gets its own Claude Code process (`claude -p`) with the
last assistant turn as context and a tight prompt: brain_update durable
knowledge or exit. It bypasses brein's own orient gate + write reminder
+ synth check via BREIN_SYNTH_SPAWNED=1 so it doesn't recurse on its
own Stop."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


_MIN_ANSWER_CHARS = 100   # skip trivial replies ("ok", "thanks")
_MAX_CONTEXT_CHARS = 6000  # keep the side-agent prompt cheap


def _write_flag_path() -> Path:
    session = os.environ.get("CLAUDE_CODE_SESSION_ID", "default")
    return Path(f"/tmp/claude-brein-write-{session}")


def _last_assistant_text(transcript_path: str) -> str:
    """Concatenate text from the trailing run of assistant messages. Skips
    metadata rows (mode/permission-mode/ai-title/attachment/hook_*/…) and
    tolerates trailing user rows."""
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    msgs = []
    for line in lines:
        try:
            msgs.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    chunks: list[str] = []
    for m in reversed(msgs):
        msg_type = m.get("type")
        role = (m.get("message") or {}).get("role") if isinstance(m.get("message"), dict) else None
        kind = role or msg_type
        if kind == "user":
            if chunks:
                break
            continue
        if kind != "assistant":
            continue
        content = (m.get("message") or {}).get("content") or m.get("content") or ""
        if isinstance(content, list):
            text = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = str(content)
        if text:
            chunks.append(text)
    return " ".join(reversed(chunks))


def _spawn_side_agent(last_answer: str) -> bool:
    """Fire-and-forget `claude -p` in a fresh session. Returns True if the
    spawn succeeded; never waits for the side-agent to finish."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False

    snippet = last_answer[-_MAX_CONTEXT_CHARS:]
    prompt = (
        "You are a brein side-agent. Below is the trailing text from another "
        "session's last assistant turn. If it contains durable knowledge "
        "(names, decisions, paths, relationships, status, contracts, people) "
        "not already in the brain, call brain_update ONCE with a short, "
        "focused doc in the matching docs/ subdir (decisions/, contacts/, "
        "companies/, projects/, knowledge/). If everything is trivial or "
        "already covered, exit without writing. Do not chat. Do not ask "
        "questions. brain_update or exit.\n\n"
        "--- last assistant turn ---\n"
        f"{snippet}\n"
        "--- end ---"
    )

    env = {**os.environ, "BREIN_SYNTH_SPAWNED": "1"}
    log_dir = Path(os.path.expanduser("~/.brein"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "synth-side-agent.log"
    try:
        log = open(log_path, "ab")  # noqa: SIM115 — intentionally not closed; child owns it
        subprocess.Popen(
            [claude_bin, "-p", prompt],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return False
    return True


def run() -> int:
    try:
        env = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return 0

    # Bypass for side-agents — they're the ones writing; they shouldn't
    # spawn their own side-agent on Stop.
    if os.environ.get("BREIN_SYNTH_SPAWNED") == "1":
        return 0
    if env.get("stop_hook_active"):
        return 0
    if _write_flag_path().exists():
        return 0

    transcript_path = env.get("transcript_path") or ""
    if not transcript_path or not Path(transcript_path).is_file():
        return 0

    last_answer = _last_assistant_text(transcript_path)
    if len(last_answer) < _MIN_ANSWER_CHARS:
        return 0

    _spawn_side_agent(last_answer)
    # Always exit 0 silently — never block the user-visible Stop.
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via CLI
    sys.exit(run())
