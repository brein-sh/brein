"""Path-safe filesystem, git, markdown parsing, keyword scoring, retrieval log,
and shared LLM invocation (CLI-first, OpenRouter fallback)."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import (
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


@contextlib.contextmanager
def _interprocess_write_lock() -> Iterator[None]:
    """Cross-process exclusive lock around the brain_update write sequence.

    Uses fcntl.flock on a sentinel file inside the repo's .git dir so that
    every brain-mcp process (each MCP stdio client spawns its own) serializes
    the full pull -> stage -> commit -> push sequence. Held across the push
    so two updates can't both win the local commit and lose the remote race.
    """
    lock_path = REPO_PATH / ".git" / "brein-write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


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


# ── LLM invocation: CLI first, OpenRouter fallback ──────────────────────────
#
# Shared by eval (one-shot judge / A-vs-B prompts) and consistency (agentic
# resolver with tool access). Both call `ask_llm`.

# Recursion guard: when a parent invokes claude/codex/gemini that itself talks
# to brain MCP, the child reads this env var to skip its own eval/consistency
# spawn, preventing a fork bomb.
LLM_GUARD_ENV = "BRAIN_EVAL_IN_PROGRESS"
_DEFAULT_CLI_PREFERENCE = "claude,codex,gemini"


def _llm_cli_preference() -> list[str]:
    return (os.environ.get("BRAIN_EVAL_CLIENT") or _DEFAULT_CLI_PREFERENCE).split(",")


def _which_llm_cli() -> str | None:
    """First available CLI in preference order, or None."""
    for name in _llm_cli_preference():
        name = name.strip()
        if name and shutil.which(name):
            return name
    return None


def _llm_cli_completion(
    cli: str,
    prompt: str,
    *,
    disable_brain: bool = False,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    timeout_s: float = 120.0,
) -> tuple[str, dict[str, Any]]:
    """Headless completion via `<cli> -p <prompt>`.

    allowed_tools: when set AND cli == "claude", passes `--allowed-tools` so
    the model can call Read/Grep/Glob/Edit. Other CLIs ignore (one-shot only).
    """
    env = os.environ.copy()
    env[LLM_GUARD_ENV] = "1"
    if disable_brain:
        env["BRAIN_EVAL_DISABLE"] = "1"
    cwd = cwd or os.path.expanduser("~")
    prompt_chars = len(prompt)

    use_json = (cli == "claude")
    cmd: list[str] = [cli, "-p", prompt]
    if use_json:
        cmd += ["--output-format", "json"]
    if allowed_tools and cli == "claude":
        cmd += ["--allowed-tools", ",".join(allowed_tools)]

    meta: dict[str, Any] = {
        "cli": cli,
        "disable_brain": disable_brain,
        "allowed_tools": list(allowed_tools) if allowed_tools else [],
        "prompt_chars": prompt_chars,
        "prompt_tokens_est": prompt_chars // 4,
        "timed_out": False,
    }
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=cwd,
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
            meta["json_parse_failed"] = True

    meta["answer_chars"] = len(text)
    meta["answer_tokens_est"] = len(text) // 4
    return text, meta


def _llm_openrouter_completion(prompt: str, *, timeout_s: float = 60.0) -> tuple[str, dict[str, Any]]:
    or_key = os.environ.get("BRAIN_EVAL_OPENROUTER_KEY", "")
    or_model = os.environ.get("BRAIN_EVAL_MODEL", "deepseek/deepseek-v4-flash")
    meta: dict[str, Any] = {
        "cli": None,
        "openrouter_model": or_model,
        "prompt_chars": len(prompt),
        "prompt_tokens_est": len(prompt) // 4,
    }
    if not or_key:
        meta["error"] = "no_openrouter_key"
        return "", meta
    body = {
        "model": or_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {or_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://brein.sh",
            "X-Title": "brain-mcp",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
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


def ask_llm(
    prompt: str,
    *,
    disable_brain: bool = False,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    timeout_s: float = 120.0,
) -> tuple[str, str, dict[str, Any]]:
    """Best available backend. Returns (text, backend_tag, meta).

    allowed_tools turns on agentic mode on claude (Read/Grep/Glob/Edit). Other
    CLIs run one-shot. OpenRouter fallback always runs one-shot.
    """
    cli = _which_llm_cli()
    if cli:
        if allowed_tools and cli == "claude":
            tag = f"cli:{cli}+agentic"
        elif disable_brain:
            tag = f"cli:{cli}+tools:no-brain"
        else:
            tag = f"cli:{cli}+tools"
        text, meta = _llm_cli_completion(
            cli, prompt,
            disable_brain=disable_brain,
            allowed_tools=allowed_tools,
            cwd=cwd,
            timeout_s=timeout_s,
        )
        return text, tag, meta
    or_model = os.environ.get("BRAIN_EVAL_MODEL", "deepseek/deepseek-v4-flash")
    text, meta = _llm_openrouter_completion(prompt, timeout_s=min(timeout_s, 60.0))
    return text, f"openrouter:{or_model}", meta
