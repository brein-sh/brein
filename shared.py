"""Path-safe filesystem, git, markdown parsing, keyword scoring, and retrieval log."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    ALLOWED_ROOT_WRITES,
    ALLOWED_WRITE_PREFIXES,
    LOG_PATH,
    REPO_PATH,
    SECRET_PATTERNS,
    STOPWORDS,
)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _run_git(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", "-C", str(REPO_PATH), *args], text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def _run_repo_cmd(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=REPO_PATH, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def _ensure_repo() -> None:
    if not (REPO_PATH / ".git").exists():
        raise RuntimeError(f"BRAIN_REPO is not a git repo: {REPO_PATH}")


def _safe_path(file_path: str) -> Path:
    if not file_path or file_path.strip() in {".", ".."}:
        raise ValueError("file_path is required")
    rel = Path(file_path)
    if rel.is_absolute():
        raise ValueError("file_path must be repo-relative")
    full = (REPO_PATH / rel).resolve()
    if full != REPO_PATH and not str(full).startswith(str(REPO_PATH) + os.sep):
        raise ValueError("file_path escapes brain repo")
    if ".git" in full.relative_to(REPO_PATH).parts:
        raise ValueError("access to .git is not allowed")
    return full


def _allowed_write_path(rel: str) -> bool:
    return rel in ALLOWED_ROOT_WRITES or rel.startswith(ALLOWED_WRITE_PREFIXES)


def _detect_secrets(text: str) -> list[str]:
    return [pat.pattern for pat in SECRET_PATTERNS if pat.search(text)]


def _frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    block = text[4:end]
    out: dict[str, Any] = {}
    for line in block.splitlines():
        if not line.strip() or line.startswith((" ", "\t", "#")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"\'')
        if value.startswith("[") and value.endswith("]"):
            value = [x.strip().strip('"\'') for x in value[1:-1].split(",") if x.strip()]
        out[key.strip()] = value
    return out


def _iter_markdown(directory: str = "."):
    base = _safe_path(directory)
    if not base.exists():
        return
    paths = base.rglob("*.md") if base.is_dir() else [base]
    for path in paths:
        if path.is_file() and ".git" not in path.relative_to(REPO_PATH).parts:
            yield path


def _tokens(query: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9_.@-]{2,}", query.lower())
    return [t for t in toks if t not in STOPWORDS]


def _line_snippets(text: str, toks: list[str], max_snippets: int = 3) -> list[dict[str, Any]]:
    lines = text.splitlines()
    snippets = []
    lower_toks = [t.lower() for t in toks]
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(t in low for t in lower_toks):
            start = max(1, i - 1)
            end = min(len(lines), i + 1)
            ctx = "\n".join(f"{n}: {lines[n-1]}" for n in range(start, end + 1))
            snippets.append({"line": i, "snippet": ctx[:1200]})
            if len(snippets) >= max_snippets:
                break
    return snippets


def _score(path: Path, text: str, fm: dict[str, Any], toks: list[str]) -> float:
    hay = text.lower()
    rel = str(path.relative_to(REPO_PATH)).lower()
    title = str(fm.get("title", path.stem)).lower()
    tags = " ".join(fm.get("tags", []) if isinstance(fm.get("tags"), list) else [str(fm.get("tags", ""))]).lower()
    score = 0.0
    for t in toks:
        count = hay.count(t)
        if count:
            score += min(count, 10)
        if t in title:
            score += 8
        if t in tags:
            score += 5
        if t in rel:
            score += 3
    if str(fm.get("source_of_truth", "")).lower() == "true":
        score += 0.5
    return score


def _truthy(value: Any) -> bool:
    return str(value).lower() == "true"


def _result_text(result: dict[str, Any]) -> dict[str, str]:
    tags = result.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    snippets = "\n".join(str(s.get("snippet", "")) for s in result.get("snippets", []) if isinstance(s, dict))
    return {
        "path": str(result.get("path", "")).lower(),
        "title": str(result.get("title", "")).lower(),
        "tags": " ".join(str(t) for t in tags).lower(),
        "snippets": snippets.lower(),
    }


def _truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _matches_filters(path: Path, fm: dict[str, Any], domain: str | None, status: str | None, tag: str | None) -> bool:
    rel_parts = path.relative_to(REPO_PATH).parts
    if domain and not (len(rel_parts) >= 2 and rel_parts[0] == "docs" and rel_parts[1] == domain):
        return False
    if status and str(fm.get("status", "")).lower() != status.lower():
        return False
    if tag:
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if tag.lower() not in [str(t).lower() for t in tags]:
            return False
    return True


def _append_retrieval_log(
    question: str,
    hits: list[str] | None,
    used_docs: list[str] | None,
    outcome: str,
    *,
    kind: str = "search",
    extra: dict[str, Any] | None = None,
) -> None:
    """Internal: append a retrieval row. Never raises into caller."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "question": question,
            "hits": hits or [],
            "used_docs": used_docs or [],
            "outcome": outcome,
        }
        if extra:
            record.update(extra)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # ponytail: log write must never break a retrieval — silent here is OK.
        pass
