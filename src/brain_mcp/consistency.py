"""Agentic consistency checker for brain writes.

When a doc is written via brain_update, a detached worker:
  1. Reads the new/updated doc.
  2. Finds the top-k semantically related existing docs via the vector index.
  3. Invokes an LLM **with tool access** (Read/Grep/Glob/Edit) to investigate
     the neighbors in full and decide:
       - ok           — independent doc, no action
       - auto_merge   — near-duplicate; agent edits the canonical doc to absorb
                        anything new + marks losers as superseded
       - supersede    — clear contradiction with a winner (newer date OR
                        source_of_truth: true); agent marks losers superseded
       - escalate     — judgment call requires a human; emit Finding only
  4. Commits and pushes any edits the agent made, under the same write lock
     brain_update uses.

Findings land in ~/.brein/consistency-queue.jsonl. Auto-resolved ones still
emit a Finding so you can audit what the agent did.
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
AGENT_TIMEOUT_SECONDS = float(os.environ.get("BRAIN_CONSISTENCY_TIMEOUT_S", "300"))
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


# --- Agentic judge (tool-enabled LLM via shared.ask_llm) ---------------------

_AGENTIC_PROMPT = """You are the consistency auditor for the company brain at:
{repo}

A doc was just written: `{new_path}`. Below is its content, followed by the
top-{k} semantically similar existing docs (snippets only — Read the full
files if you need to decide).

Your job: investigate, then either RESOLVE or ESCALATE.

You have these tools: Read, Grep, Glob, Edit. Use them.

Decide one of:

  - "ok"          — new doc is independent / complementary. Do nothing.
  - "auto_merge"  — near-duplicate facts exist. Use Edit to absorb anything
                    new from the loser(s) into the canonical doc, then on
                    each deprecated file insert this line as the first line
                    after the frontmatter `---` close:

                        > **Superseded by [[canonical-doc-title]].**

  - "supersede"   — contradiction with a CLEAR winner. The winner is the doc
                    with `source_of_truth: true`, or (if both/neither have it)
                    the one with the newer `last_reviewed` / `decided` date.
                    Use Edit to insert the superseded line on each loser.

  - "escalate"    — contradiction with no clear winner, or you're not sure.
                    DO NOT EDIT. Output an escalation_reason explaining what
                    a human needs to decide.

Rules:
  - Default to "ok" when uncertain. Never edit on a guess.
  - "auto_merge" / "supersede" require confidence="high" AND a single clear
    canonical_path. Otherwise escalate.
  - Never touch .git/, never edit files outside {repo}/docs/.
  - Be surgical: do NOT rewrite full files. Insert the supersede line only.
    For auto_merge, append truly-new facts into the canonical doc as bullets.
  - Preserve original frontmatter exactly.

When done, output ONE JSON object (no prose, no markdown fence):

{{
  "kind": "ok" | "auto_merge" | "supersede" | "escalate",
  "confidence": "high" | "medium" | "low",
  "summary": "one-line plain English",
  "canonical_path": "docs/..." or null,
  "deprecated_paths": ["docs/...", ...],
  "edits_applied": true or false,
  "escalation_reason": "why a human is needed" or null
}}

--- NEW DOC: {new_path} ---
{new_content}

--- NEIGHBORS (snippets; Read for full text) ---
{neighbors_block}
""".rstrip()


def _judge_agentic(rel: str, new_content: str, neighbors: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Invoke the agentic judge. Returns parsed JSON or None on failure."""
    from .shared import ask_llm

    neighbor_blocks = []
    for n in neighbors:
        path = n.get("path", "?")
        snippet = n.get("vector_snippet") or ""
        neighbor_blocks.append(
            f"--- NEIGHBOR: {path} (vec_score={n.get('vector_score', 0):.3f}) ---\n{snippet}"
        )
    neighbors_block = "\n\n".join(neighbor_blocks) if neighbor_blocks else "(none)"

    prompt = _AGENTIC_PROMPT.format(
        repo=str(REPO_PATH),
        new_path=rel,
        new_content=new_content[:4000],
        k=len(neighbors),
        neighbors_block=neighbors_block[:8000],
    )

    text, _backend, _meta = ask_llm(
        prompt,
        disable_brain=True,
        allowed_tools=["Read", "Grep", "Glob", "Edit"],
        cwd=str(REPO_PATH),
        timeout_s=AGENT_TIMEOUT_SECONDS,
    )
    return _extract_json(text)


def _commit_agent_edits(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Stage, commit, and push edits the agent made. Returns commit info or
    None if nothing was actually changed. Uses the same write lock brain_update
    uses so we serialize with concurrent writers."""
    from .shared import _interprocess_write_lock, _run_git

    try:
        with _interprocess_write_lock():
            status = _run_git(["status", "--porcelain"])
            if not (status.stdout or "").strip():
                return None
            _run_git(["add", "-A"])
            kind = payload.get("kind", "ok")
            summary = (payload.get("summary") or "")[:60]
            canonical = payload.get("canonical_path") or ""
            msg = f"consistency({kind}): {summary}"
            if canonical:
                msg += f" → {canonical}"
            commit = _run_git(["commit", "--quiet", "-m", msg])
            if commit.returncode != 0:
                return None
            _run_git(["push", "--quiet", "origin", "HEAD"])
            sha = _run_git(["rev-parse", "HEAD"]).stdout.strip()
            return {"sha": sha, "message": msg}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


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

_KIND_MAP: dict[str, FindingKind] = {
    "auto_merge": "auto_merge",
    "supersede": "contradiction",
    "escalate": "unresolved",
    "ok": "ok",
}


def run_check(write_path: str) -> Finding | None:
    """Foreground consistency check for one written doc. Returns the emitted
    Finding (or None if 'ok' with no edits / index not ready / no neighbors)."""
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
        return None

    payload = _judge_agentic(rel, new_content, neighbors)
    judge = "agentic" if payload else "stub"
    if payload is None:
        # Agent unavailable / output unparseable. Emit a low-confidence
        # unresolved finding so the queue at least records the attempt.
        payload = {
            "kind": "escalate",
            "confidence": "low",
            "summary": f"agentic judge unavailable; {len(neighbors)} similar docs found",
            "canonical_path": None,
            "deprecated_paths": [n["path"] for n in neighbors],
            "edits_applied": False,
            "escalation_reason": "judge_unavailable",
        }

    raw_kind = str(payload.get("kind", "ok"))
    edits_applied = bool(payload.get("edits_applied"))
    commit_info = _commit_agent_edits(payload) if edits_applied else None

    if raw_kind == "ok" and not edits_applied:
        return None  # silent

    finding = Finding(
        finding_id=uuid.uuid4().hex[:12],
        write_path=rel,
        kind=_KIND_MAP.get(raw_kind, "unresolved"),
        confidence=str(payload.get("confidence", "low")),
        summary=str(payload.get("summary", "")),
        suggested_fix=payload.get("escalation_reason") or payload.get("canonical_path"),
        related_paths=list(payload.get("deprecated_paths", []) or [])[:10],
        created_at=_now_iso(),
        judge=judge,
        raw={**payload, "commit": commit_info} if commit_info else payload,
    )
    append_finding(finding)
    return finding


# --- detached spawn -----------------------------------------------------------

def spawn_detached(write_path: str) -> int:
    """Launch the consistency check as a detached subprocess. Returns the pid.

    Uses `python -m brain_mcp.cli consistency check <path>` via the running
    interpreter — PATH-independent so launchd's minimal env doesn't break us.
    Same pattern as eval._spawn_eval_worker.
    """
    log_path = QUEUE_PATH.with_name("consistency-worker.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    cmd = [sys.executable, "-m", "brain_mcp.cli", "consistency", "check", write_path]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid
