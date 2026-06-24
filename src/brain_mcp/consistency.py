"""Background consistency checker for brain writes.

When a doc is written via brain_update, a detached worker:
  1. Reads the new/updated doc.
  2. Finds the top-k semantically related existing docs via the vector
     index (no extra LLM call for this step).
  3. Asks an LLM judge whether the new doc contradicts or duplicates
     anything in the neighbors. Structured JSON output.
  4. Applies an action ladder:
       - confidence=high & duplicate detected   → emit `auto_merge` finding
       - confidence=high & contradiction        → emit `contradiction` finding (user-visible)
       - confidence=low                         → emit `unresolved` finding (GitHub-issue-worthy)
       - no drift                                → silent, no finding

Findings land in ~/.brein/consistency-queue.jsonl (append-only). Agents
pull pending findings via the brain_consistency_status MCP tool.

This module is the v1 — auto-correction and GitHub-issue filing are
left to follow-up iterations. Today the worker only EMITS findings;
the agent / user chooses what to do with them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import REPO_PATH, VECTOR_INDEX_PATH


QUEUE_PATH = VECTOR_INDEX_PATH.with_name("consistency-queue.jsonl")
HERMES_TIMEOUT_SECONDS = 60.0
TOP_K_NEIGHBORS = 5
SIMILAR_THRESHOLD = 0.80  # below this we don't even bother judging


FindingKind = Literal["auto_merge", "contradiction", "unresolved", "ok"]


@dataclass(frozen=True)
class Finding:
    finding_id: str
    write_path: str
    kind: FindingKind
    confidence: str  # high | medium | low
    summary: str
    suggested_fix: str | None
    related_paths: list[str]
    created_at: str
    judge: str  # which judge produced it (e.g. "hermes" or "stub")
    raw: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_finding(finding: Finding) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(finding.to_json()) + "\n")


def read_queue() -> list[Finding]:
    if not QUEUE_PATH.exists():
        return []
    out: list[Finding] = []
    for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(Finding(**d))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def clear_queue() -> int:
    if not QUEUE_PATH.exists():
        return 0
    n = sum(1 for _ in QUEUE_PATH.read_text(encoding="utf-8").splitlines())
    QUEUE_PATH.unlink()
    return n


# --- top-k neighbor lookup ----------------------------------------------------

def _find_neighbors(write_path: str, text: str, k: int = TOP_K_NEIGHBORS) -> list[dict[str, Any]]:
    """Use the existing vector index to find docs similar to `text`.

    No re-embed of the full corpus — just the new doc. If the index isn't
    ready we return [] and the judge skips. Falls back to grep-of-first-line
    for the keyword fallback to keep the worker useful even with no index.
    """
    from . import vector  # local import: vector loads fastembed lazily

    try:
        hits = vector._best_vector_hits(
            query=text[:2000],
            directory="docs",
            domain=None,
            tag=None,
            status=None,
            limit=k + 1,  # +1 because the new doc itself may rank #1
            force_rebuild=False,
        )[0]
    except Exception:
        return []
    # Strip the write_path itself if it's in the results (self-match).
    return [h for h in hits if h.get("path") != write_path][:k]


# --- LLM judge (Hermes one-shot) ---------------------------------------------

_JUDGE_PROMPT = """You are auditing a knowledge brain for internal consistency.

A new or updated doc has just been written. Below is its content, followed
by the top-K most semantically similar existing docs. Decide whether the
new doc:

  - DUPLICATES an existing doc (same fact, restated)
  - CONTRADICTS an existing doc (different fact about the same entity)
  - is OK (independent, no overlap or fully complementary)

Respond with ONE JSON object, no prose:

{{
  "kind": "auto_merge" | "contradiction" | "unresolved" | "ok",
  "confidence": "high" | "medium" | "low",
  "summary": "one-line plain English description of the finding",
  "suggested_fix": "if contradiction or duplicate, what should change (else null)",
  "related_paths": ["docs/...", "docs/..."]
}}

Rules:
  - "ok" if the new doc is independent. Always allowed.
  - "auto_merge" only if a near-identical fact already exists.
  - "contradiction" only if facts disagree (dates, names, values, decisions).
  - "unresolved" if you can SEE there's a potential conflict but you're not sure.
  - Use "low" confidence freely — better to surface uncertainty than auto-fix wrong.

--- NEW DOC: {new_path} ---
{new_content}

--- NEIGHBORS ---
{neighbors_block}
""".rstrip()


def _judge_bin() -> tuple[str, list[str]] | None:
    """Return (binary, extra_args) for the judge. Override via
    BRAIN_JUDGE_CMD env (full command, prompt appended). Defaults to
    `claude -p ... --output-format text` since Claude Code is the most
    likely installed agent CLI."""
    custom = os.environ.get("BRAIN_JUDGE_CMD", "").strip()
    if custom:
        import shlex
        parts = shlex.split(custom)
        return parts[0], parts[1:]
    claude = shutil.which("claude")
    if claude:
        return claude, ["-p", None, "--output-format", "text"]  # None = prompt slot
    return None


def _judge_with_llm(new_path: str, new_content: str, neighbors: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Run the judge prompt through the configured LLM CLI. Returns
    parsed JSON or None on failure."""
    bin_info = _judge_bin()
    if not bin_info:
        return None
    binpath, extra = bin_info

    neighbor_blocks = []
    for n in neighbors:
        rel = n.get("path", "?")
        snippet = n.get("vector_snippet") or ""
        neighbor_blocks.append(f"--- NEIGHBOR: {rel} (vec_score={n.get('vector_score', 0):.3f}) ---\n{snippet}")
    neighbors_block = "\n\n".join(neighbor_blocks) if neighbor_blocks else "(none)"

    prompt = _JUDGE_PROMPT.format(
        new_path=new_path,
        new_content=new_content[:4000],
        neighbors_block=neighbors_block[:8000],
    )

    argv = [binpath, *[(prompt if a is None else a) for a in extra]] if None in extra else [binpath, *extra, prompt]

    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=HERMES_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None

    return _extract_json(completed.stdout)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    # Try the whole thing first; then find a balanced {...} block.
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


# --- runner -------------------------------------------------------------------

def run_check(write_path: str) -> Finding | None:
    """Foreground consistency check for one written doc. Returns the
    emitted Finding (or None if 'ok' / index not ready / no neighbors)."""
    abs_path = (REPO_PATH / write_path).resolve() if not Path(write_path).is_absolute() else Path(write_path)
    if not abs_path.exists():
        return None
    try:
        new_content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    rel = str(abs_path.relative_to(REPO_PATH)) if str(abs_path).startswith(str(REPO_PATH)) else str(abs_path)

    neighbors = _find_neighbors(rel, new_content, k=TOP_K_NEIGHBORS)
    neighbors = [n for n in neighbors if n.get("vector_score", 0) >= SIMILAR_THRESHOLD]
    if not neighbors:
        # No semantically nearby docs — nothing to check.
        return None

    payload = _judge_with_llm(rel, new_content, neighbors)
    judge = "llm" if payload else "stub"
    if payload is None:
        # LLM judge unavailable. Emit a low-confidence "unresolved" finding
        # so the agent at least knows we tried.
        payload = {
            "kind": "unresolved",
            "confidence": "low",
            "summary": f"LLM judge unavailable; {len(neighbors)} similar docs were found",
            "suggested_fix": None,
            "related_paths": [n["path"] for n in neighbors],
        }

    kind = payload.get("kind", "ok")
    if kind == "ok":
        return None  # silent

    finding = Finding(
        finding_id=uuid.uuid4().hex[:12],
        write_path=rel,
        kind=kind,
        confidence=str(payload.get("confidence", "low")),
        summary=str(payload.get("summary", "")),
        suggested_fix=payload.get("suggested_fix"),
        related_paths=list(payload.get("related_paths", []) or [])[:10],
        created_at=_now_iso(),
        judge=judge,
        raw=payload,
    )
    append_finding(finding)
    return finding


# --- detached spawn -----------------------------------------------------------

def spawn_detached(write_path: str) -> int:
    """Launch `brein consistency check <path>` as a detached background
    process. Returns the pid. Does not wait."""
    brein = _brein_executable()
    log_path = QUEUE_PATH.with_name("consistency-worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [brein, "consistency", "check", write_path],
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid


def _brein_executable() -> str:
    cand = sys.argv[0] if sys.argv and sys.argv[0] else None
    if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
        return cand
    return shutil.which("brein") or "brein"
