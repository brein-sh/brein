"""Auto-logging decorator for MCP tool calls.

Emits one JSONL event per tool call to LOG_PATH. Field names follow
OpenTelemetry GenAI semconv so events drop into Phoenix / Langfuse / OTLP
backends later without rewriting the schema.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .config import LOG_PATH, REPO_PATH

_FLUSH_EVERY = int(os.environ.get("BRAIN_TELEMETRY_FLUSH_EVERY", "10"))
_event_count = 0
_count_lock = threading.Lock()


def _maybe_git_flush() -> None:
    """Auto-commit-and-push the telemetry log every _FLUSH_EVERY events.

    Runs in a daemon thread so MCP tool latency is unaffected. Silently
    swallows all git errors — telemetry must never break the tool call.
    Skipped unless LOG_PATH lives inside REPO_PATH (so we never try to
    git-add a file outside the repo).
    """
    global _event_count
    try:
        log_rel = LOG_PATH.resolve().relative_to(REPO_PATH.resolve())
    except ValueError:
        return  # log is outside the repo; nothing to flush
    with _count_lock:
        _event_count += 1
        if _event_count % _FLUSH_EVERY != 0:
            return
        count = _event_count
    threading.Thread(target=_git_flush, args=(str(log_rel), count), daemon=True).start()


def _git_flush(rel_path: str, count: int) -> None:
    def run(args: list[str]) -> int:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_PATH), *args],
                text=True,
                capture_output=True,
                timeout=30,
            ).returncode
        except Exception:
            return 1

    try:
        if run(["add", rel_path]) != 0:
            return
        # Only commit if there's actually staged content.
        if run(["diff", "--cached", "--quiet", "--", rel_path]) == 0:
            return
        if run(["commit", "-m", f"telemetry: {count} events", "--", rel_path]) != 0:
            return
        run(["push", "origin", "main"])  # best-effort; next flush retries on failure
    except Exception:
        pass

# ponytail: deny-list misses new sensitive keys; substring match keeps it short.
_SENSITIVE_SUBSTRINGS = ("api_key", "token", "password", "private_key", "seed", "secret", "passwd", "auth")
_LONG_STRING_CAP = 500


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if any(s in k.lower() for s in _SENSITIVE_SUBSTRINGS):
            out[k] = "***"
        elif isinstance(v, str) and len(v) > _LONG_STRING_CAP:
            out[k] = v[:_LONG_STRING_CAP] + "…"
        else:
            out[k] = v
    return out


def _query_hash(args: dict[str, Any]) -> str | None:
    for key in ("query", "question", "text", "file_path"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return "sha256:" + hashlib.sha256(v.strip().lower().encode("utf-8")).hexdigest()[:16]
    return None


def _write_event(record: dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        _maybe_git_flush()
    except Exception:
        # ponytail: telemetry must never break the tool call.
        pass


def logged(tool_name: str | None = None) -> Callable:
    """Wrap an MCP tool so every invocation emits a tool_call event.

    Apply *under* @mcp.tool so FastMCP registers the wrapper:
        @mcp.tool(name="foo")
        @logged("foo")
        def foo(...): ...
    """
    def decorator(fn: Callable) -> Callable:
        resolved = tool_name or fn.__name__
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                bound = sig.bind_partial(*args, **kwargs)
                arg_dict = dict(bound.arguments)
            except TypeError:
                arg_dict = dict(kwargs)
                for i, v in enumerate(args):
                    arg_dict[f"_arg{i}"] = v
            redacted = _redact(arg_dict)
            ok = True
            err: str | None = None
            result_chars = 0
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, str):
                    result_chars = len(result)
                return result
            except Exception as exc:
                ok = False
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _write_event({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "kind": "tool_call",
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": resolved,
                    "mcp.method.name": "tools/call",
                    "args": redacted,
                    "query_hash": _query_hash(arg_dict),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "ok": ok,
                    "error": err,
                    "result_chars": result_chars,
                })
        return wrapper
    return decorator
