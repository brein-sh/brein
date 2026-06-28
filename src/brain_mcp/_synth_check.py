"""Stop-hook backend: scan the last assistant turn for proper-noun-ish
entities the brain doesn't cover, and emit a Claude Code block-reason
nudging the model to brain_update before stopping.

Wired from cli.py as `brein eval check-synthesis`. Reads the Stop-hook
JSON envelope on stdin, prints a block JSON on stdout iff there's a
real gap and no brain_update happened this turn. Stays silent on
trivial turns, missing transcripts, missing brain, or repeat fires
(stop_hook_active=True)."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .config import REPO_PATH


_MIN_ANSWER_CHARS = 400
_MAX_GAPS_REPORTED = 8

# Tokens that look proper-noun-ish but are too common to be a real entity.
_NOISE_TOKENS = {
    # Common English Capitalized words
    "The", "This", "That", "With", "From", "Your", "Have", "Need", "Want",
    "Like", "Each", "Every", "Most", "Some", "Will", "Should", "Could",
    "Would", "After", "Before", "While", "About", "Brain", "Memory",
    "Here", "There", "Just", "Only", "Both", "Also", "Then", "Even",
    "Make", "Made", "Said", "Says", "Goes", "Such", "More", "Less",
    "When", "Where", "Which", "Whom", "What", "Note", "True", "False",
    "None", "Null", "TODO", "FIXME", "Also", "Same",
    # Generic protocol / format / spec acronyms — not entities.
    "API", "APIs", "HTTP", "HTTPS", "URL", "URLs", "URI", "JSON", "YAML",
    "XML", "CSV", "TSV", "SQL", "REST", "RPC", "gRPC", "GraphQL",
    "OpenAPI", "OAuth", "JWT", "CORS", "TLS", "SSL", "SSH", "DNS",
    "TCP", "UDP", "IP", "UTF", "ASCII", "UUID", "GUID", "MIME",
    # Common framework / tool nouns — usually mentioned in passing.
    "CLAUDE", "README", "MCP", "CLI", "GUI", "SDK", "SDKs",
    "Bearer", "Express", "Node", "Python", "TypeScript", "JavaScript",
    "Rust", "Bash", "Docker", "GitHub", "GitLab", "Git",
    # ANSI / box-drawing artifacts that occasionally tokenize weirdly.
    "ANSI",
}


def _last_assistant_text(transcript_path: str) -> str:
    """Concatenate text blocks from the most recent contiguous run of
    assistant messages. Claude Code transcripts intersperse a lot of
    metadata (mode, permission-mode, ai-title, last-prompt, attachment,
    file-history-snapshot, hook_*…) between role-bearing rows — those are
    all ignored. Trailing `user` rows are also tolerated: we keep walking
    past them until we find the assistant run, then stop when we hit the
    `user` that *preceded* that run."""
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
        # In Claude Code transcripts, the top-level `type` is "assistant" /
        # "user" / etc., and the message itself sits under "message". Other
        # types ("attachment", "system", "mode", "ai-title", …) are metadata.
        msg_type = m.get("type")
        role = (m.get("message") or {}).get("role") if isinstance(m.get("message"), dict) else None
        kind = role or msg_type
        if kind == "user":
            if chunks:
                break  # we already have the trailing assistant run
            continue   # trailing user/metadata — keep going to find assistant
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


def _candidate_entities(text: str) -> set[str]:
    """Proper-noun-ish tokens: ALL-CAPS acronyms, dotted/dashed identifiers,
    CamelCase, or Capitalized words of 4+ chars. Filter obvious noise."""
    out: set[str] = set()
    for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9._-]{2,}\b", text):
        if tok in _NOISE_TOKENS:
            continue
        if "-" in tok or "." in tok:
            out.add(tok)
        elif len(tok) >= 3 and tok.isupper():
            out.add(tok)
        elif len(tok) >= 4 and any(c.isupper() for c in tok[1:]):
            out.add(tok)  # CamelCase / mixedCase
        elif len(tok) >= 4 and tok[0].isupper() and tok[1:].islower():
            out.add(tok)  # Capitalized
    return out


def _docs_blob(docs_dir: Path) -> str:
    parts: list[str] = []
    for root, _dirs, files in os.walk(docs_dir):
        for f in files:
            if not f.lower().endswith((".md", ".markdown", ".txt")):
                continue
            try:
                parts.append(Path(root, f).read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(parts)


def _write_flag_path() -> Path:
    session = os.environ.get("CLAUDE_CODE_SESSION_ID", "default")
    return Path(f"/tmp/claude-brein-write-{session}")


def run() -> int:
    """Entry point. Returns process exit code; prints decision JSON on stdout."""
    try:
        env = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return 0

    # Avoid loops — Claude Code sets this when a Stop hook already fired
    # for this turn and the model is trying to stop again.
    if env.get("stop_hook_active"):
        return 0

    # If brain_update already happened this turn, the model met its duty.
    if _write_flag_path().exists():
        return 0

    transcript_path = env.get("transcript_path") or ""
    if not transcript_path or not Path(transcript_path).is_file():
        return 0

    brain_repo = Path(os.environ.get("BRAIN_REPO") or REPO_PATH)
    docs_dir = brain_repo / "docs"
    if not docs_dir.is_dir():
        return 0

    answer = _last_assistant_text(transcript_path)
    if len(answer) < _MIN_ANSWER_CHARS:
        return 0

    candidates = _candidate_entities(answer)
    if not candidates:
        return 0

    blob = _docs_blob(docs_dir)
    gaps = sorted(t for t in candidates if t not in blob)
    if not gaps:
        return 0

    shown = gaps[:_MAX_GAPS_REPORTED]
    reason = (
        f"Your last turn synthesized info about: {', '.join(shown)}. "
        f"None of these names appear anywhere in {docs_dir}. "
        "If they're real entities the user works with (companies, products, repos, contracts, people), "
        "call brain_update now — one short doc per entity, in the matching docs/ subdir "
        "(decisions/, contacts/, companies/, projects/, knowledge/). "
        "If they're just code identifiers, library names, or other noise, "
        "stop again and this hook will let you go."
    )
    sys.stdout.write(json.dumps({"decision": "block", "reason": reason}) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via CLI
    sys.exit(run())
