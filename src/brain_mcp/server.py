#!/usr/bin/env python3
"""Policy-aware MCP server for an org-wide knowledge brain.

Exposes curated retrieval + safe write tools over stdio. Designed for Hermes,
Claude Code, Codex, or any MCP client.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from importlib import resources
from typing import Any

from mcp.server.fastmcp import FastMCP


def _bundled_script(name: str, env_override: str) -> str:
    """Return the absolute path to a bundled brein script, with env override.

    `name` is the file under `brain_mcp/_scripts/`. `env_override` lets users
    point at a custom version in their own brain repo. Without an override,
    we ship the script as part of the package — so a fresh markdown brain
    repo never needs to provide its own `scripts/` directory.
    """
    custom = os.environ.get(env_override)
    if custom:
        return custom
    return str(resources.files("brain_mcp._scripts").joinpath(name))

from .config import (
    EMBEDDING_MODEL_NAME,
    HYBRID_KEYWORD_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    LOG_PATH,
    MAX_READ_CHARS,
    REPO_PATH,
    RERANK_MAX_TOP_K,
    VECTOR_INDEX_PATH,
)
from . import consistency, index_state, index_worker
from .rerank import _maybe_rerank
from .telemetry import logged
from .shared import (
    _allowed_write_path,
    _append_retrieval_log,
    _detect_secrets,
    _ensure_repo,
    _frontmatter,
    _interprocess_write_lock,
    _iter_markdown,
    _json,
    _line_snippets,
    _matches_filters,
    _result_text,  # noqa: F401  (re-exported for backwards compat)
    _run_git,
    _run_repo_cmd,
    _safe_path,
    _score,
    _tokens,
)
from .vector import _best_vector_hits, _get_embedder_backend, _load_vector_index, _vector_health
from .eval import maybe_eval

mcp = FastMCP(
    "Brain",
    instructions=(
        "A policy-aware brain that lives in a git repo. Use for company/team-shareable "
        "knowledge only. Keep personal/private facts out. Never store secrets. "
        "Search/read before writing; writes validate, regenerate index, commit, and push."
    ),
)


def _pull_ff() -> None:
    _run_git(["pull", "--ff-only", "origin", "main"], check=True)


def _validate_after_write(changed_docs: bool) -> list[str]:
    messages = []
    index_script = _bundled_script("generate_index.py", "BRAIN_INDEX_SCRIPT")
    validate_script = _bundled_script("validate_docs.py", "BRAIN_VALIDATE_SCRIPT")
    if changed_docs:
        _run_repo_cmd(["python3", index_script], check=True)
        messages.append("regenerated docs/index.md")
    _run_repo_cmd(["python3", validate_script], check=True)
    messages.append("validate_docs passed")
    _run_repo_cmd(["python3", index_script, "--check"], check=True)
    messages.append("generated index check passed")
    return messages


def _restore_paths(paths: list[str], created: list[str]) -> None:
    """Best-effort rollback: restore tracked paths from HEAD, delete brand-new files."""
    for rel in created:
        full = REPO_PATH / rel
        try:
            if full.exists():
                full.unlink()
        except OSError:
            pass
    tracked = [p for p in paths if p not in created]
    if tracked:
        _run_git(["checkout", "--", *tracked])
    _run_git(["reset", "HEAD", "--", *paths])


# ponytail: shared lock so async pushes (from update) and telemetry flushes
# don't race a non-ff into each other at the wire.
_push_lock = threading.Lock()


def _bg_push() -> None:
    with _push_lock:
        r = _run_git(["push", "origin", "main"])
        if r.returncode != 0:
            print(
                f"[brain-mcp] async push failed: {(r.stderr or r.stdout).strip()[:200]}",
                file=sys.stderr,
                flush=True,
            )


def _commit_push(paths: list[str], commit_message: str) -> dict[str, Any]:
    _run_git(["add", *paths], check=True)
    diff_cached = _run_git(["diff", "--cached", "--quiet"])
    if diff_cached.returncode == 0:
        return {"changed": False, "message": "No changes to commit."}
    _run_git(["commit", "-m", commit_message], check=True)
    status = _run_git(["status", "--short"], check=True).stdout.strip()
    head = _run_git(["rev-parse", "--short", "HEAD"], check=True).stdout.strip()
    # Push synchronously while the caller still holds the inter-process write
    # lock — otherwise two writers could both win their local commit and then
    # race the remote non-ff. Per-process _push_lock still gates the rare
    # telemetry/async pushes that happen outside brain_update.
    with _push_lock:
        push = _run_git(["push", "origin", "main"])
    if push.returncode != 0:
        return {
            "changed": True,
            "pushed": "failed",
            "push_error": (push.stderr or push.stdout).strip()[:400],
            "clean": status == "",
            "dirty_status": status,
            "head": head,
        }
    return {"changed": True, "pushed": "ok", "clean": status == "", "dirty_status": status, "head": head}


def _index_status_payload() -> str | None:
    """Return a status payload if the index isn't usable; None if it's ready.

    Auto-spawns the background worker when status is missing or stalled, so
    the agent only needs to grep until the next call returns 'ready'. The
    spawn is fire-and-forget; we don't wait.
    """
    status, state = index_state.resolve_status()
    if status == "ready":
        return None

    auto_spawned = False
    if status in {"missing", "stalled"}:
        try:
            index_worker.spawn_detached()
            auto_spawned = True
            status, state = index_state.resolve_status()
        except Exception as exc:  # spawning is best-effort
            return _json({
                "status": "stalled",
                "action": "use_grep",
                "hint": f"Could not spawn index worker ({exc}). Use Grep/Read over {REPO_PATH} until brain_search returns status=ready.",
                "repo_path": str(REPO_PATH),
                "last_error": getattr(state, "last_error", None),
            })

    progress = None
    if state and state.total:
        progress = f"{state.done}/{state.total} ({int(100 * state.done / state.total)}%)"

    payload = {
        "status": status,
        "action": "use_grep",
        "hint": (
            f"Index is {status}. Use Grep/Read over {REPO_PATH}/docs until "
            f"brain_search returns status=ready. Call brain_search again later to retry."
        ),
        "repo_path": str(REPO_PATH),
        "progress": progress,
        "auto_spawned_worker": auto_spawned,
    }
    if state and state.last_error:
        payload["last_error"] = state.last_error.splitlines()[0]
    return _json(payload)


@mcp.tool(name="brain_search")
@logged("brain_search")
def brain_search(
    query: str,
    directory: str = "docs",
    domain: str | None = None,
    tag: str | None = None,
    status: str | None = None,
    max_results: int = 10,
    rerank: bool = False,
    rerank_top_k: int = 25,
    rerank_method: str = "llm",
) -> str:
    """Semantic brain search via embeddings.

    Returns ranked vector hits over the configured brain repo. brain_search
    is embeddings-only — for literal/keyword lookup, agents should use their
    normal Grep/Read tools over $BRAIN_REPO directly.

    If the index isn't ready (missing, building, stalled, empty), returns a
    structured status payload with action='use_grep' instead of degraded
    results. The agent should fall back to Grep over the repo until status
    flips to 'ready'. A missing/stalled index auto-spawns a background
    builder; the agent never has to wait.
    """
    _ensure_repo()
    if not query.strip():
        return _json({"error": "query is required"})
    if rerank_method not in {"llm", "heuristic"}:
        rerank_method = "llm"

    gate = _index_status_payload()
    if gate is not None:
        return gate

    toks = _tokens(query) or [query.lower()]
    vector_results, vector_meta = _best_vector_hits(
        query=query,
        directory=directory,
        domain=domain,
        tag=tag,
        status=status,
        limit=max(max_results * 4, min(rerank_top_k if rerank else 25, RERANK_MAX_TOP_K), 25),
        force_rebuild=False,
    )
    vector_meta["enabled"] = True

    results = []
    for r in vector_results:
        snippets = [r.get("vector_snippet")] if r.get("vector_snippet") else []
        results.append({
            "path": r["path"],
            "score": round(r["vector_score"], 6),
            "vector_score": round(r["vector_score"], 6),
            "title": r.get("title"),
            "status": r.get("status"),
            "tags": r.get("tags", []),
            "source_of_truth": r.get("source_of_truth"),
            "snippets": snippets,
            "retrieval": "vector",
        })
    results, rerank_meta = _maybe_rerank(query, toks, results, rerank, rerank_method, rerank_top_k)
    top = results[:max_results]
    _append_retrieval_log(
        query, [r["path"] for r in top], None, "search_vector",
        kind="search", extra={"backend": vector_meta.get("backend")},
    )
    # Fire-and-forget LLM-gated A/B eval. Spawned as a detached subprocess
    # in maybe_eval, so it survives this MCP server's exit. Skipped if the
    # gate decides the query isn't significant, or if we've already evaluated
    # this query_hash in the last 24h. See src/brain_mcp/eval.py.
    evidence_preview = "\n".join(
        f"- {r['path']} (score={r.get('score'):.3f})" for r in top
    )
    maybe_eval(question=query, evidence_block=evidence_preview)
    return _json({
        "status": "ready",
        "query": query,
        "tokens": toks,
        "vector": vector_meta,
        "rerank": rerank_meta,
        "results": top,
        "truncated": len(results) > max_results,
    })


@mcp.tool(name="brain_index_status")
@logged("brain_index_status")
def brain_index_status(restart_if_stalled: bool = False, force_rebuild: bool = False) -> str:
    """Inspect and optionally restart the brain's vector index builder.

    Use this when brain_search returned status != 'ready' and you want to
    know how far along the build is, or kick off a fresh build:

      restart_if_stalled=True   spawn a new worker iff status is 'stalled'
                                or 'missing'. No-op if already 'building'.
      force_rebuild=True        kill any existing worker and start fresh.
                                Use sparingly; full rebuild is expensive.

    Returns the resolved status + progress + repo path. Agents should
    keep using Grep/Read over the repo while status != 'ready'.
    """
    import os
    import signal

    status, state = index_state.resolve_status()

    if force_rebuild:
        if state and state.worker_pid:
            try:
                os.kill(state.worker_pid, signal.SIGTERM)
            except OSError:
                pass
        index_state.clear()
        pid = index_worker.spawn_detached()
        return _json({
            "status": "building",
            "worker_pid": pid,
            "action": "use_grep",
            "repo_path": str(REPO_PATH),
            "note": "fresh rebuild started; previous worker terminated if any",
        })

    if restart_if_stalled and status in {"stalled", "missing"}:
        pid = index_worker.spawn_detached()
        status, state = index_state.resolve_status()
        return _json({
            "status": status,
            "worker_pid": pid,
            "action": "use_grep" if status != "ready" else "search_now",
            "repo_path": str(REPO_PATH),
            "progress": _progress_str(state),
            "note": "background worker spawned",
        })

    return _json({
        "status": status,
        "action": "search_now" if status == "ready" else "use_grep",
        "repo_path": str(REPO_PATH),
        "progress": _progress_str(state),
        "worker_pid": state.worker_pid if state else None,
        "last_error": (state.last_error.splitlines()[0] if state and state.last_error else None),
    })


def _progress_str(state) -> str | None:
    if not state or not state.total:
        return None
    return f"{state.done}/{state.total} ({int(100 * state.done / state.total)}%)"


# Internal helper — used by brain_evidence. No longer exposed as an MCP tool;
# agents should use their normal Read/Glob/Grep tools against the brain repo.
@logged("brain_read")
def brain_read(file_path: str) -> str:
    """Read a specific brain file by repo-relative path."""
    _ensure_repo()
    full = _safe_path(file_path)
    if not full.exists() or not full.is_file():
        return _json({"error": f"not found: {file_path}"})
    content = full.read_text(encoding="utf-8")
    rel = str(full.relative_to(REPO_PATH))
    _append_retrieval_log(file_path, [rel], [rel], "read", kind="read")
    return _json({
        "path": rel,
        "frontmatter": _frontmatter(content),
        "content": content[:MAX_READ_CHARS],
        "truncated": len(content) > MAX_READ_CHARS,
    })


@mcp.tool(name="brain_evidence")
@logged("brain_evidence")
def brain_evidence(question: str, max_docs: int = 5) -> str:
    """Return an evidence bundle for a question: ranked docs + snippets + citations.

    This tool does NOT synthesize a final natural-language answer. It returns
    the docs the client agent should use to write one, with paths to cite.

    Use brain_evidence when you have a question and want grounded context in
    one round-trip (search + read of top hits). Use brain_search when you only
    need to know which docs exist on a topic.
    """
    search_raw = brain_search(question, max_results=max_docs)
    search = json.loads(search_raw)
    used = []
    for result in search.get("results", []):
        read = json.loads(brain_read(result["path"]))
        content = read.get("content", "")
        used.append({
            "path": result["path"],
            "title": result.get("title"),
            "score": result.get("score"),
            "snippets": result.get("snippets", []),
            "frontmatter": read.get("frontmatter", {}),
            "excerpt": content[:2500],
        })
    _append_retrieval_log(question, [u["path"] for u in used], [u["path"] for u in used], "evidence_bundle", kind="answer")
    # ponytail: kick off a non-blocking background A/B if eval is enabled and
    # this query trips a trigger. Fire-and-forget; never blocks the client.
    evidence_block = "\n\n".join(
        f"--- {u['path']} ---\n{u['excerpt']}" for u in used
    )
    maybe_eval(question=question, evidence_block=evidence_block)
    return _json({"question": question, "evidence": used, "answer_instruction": "Use only this evidence unless you call more tools. Cite paths in the final answer."})


@mcp.tool(name="brain_update")
@logged("brain_update")
def brain_update(file_path: str, content: str, commit_message: str, mode: str = "replace") -> str:
    """Create/replace/append a curated brain file, validate, commit, and push."""
    _ensure_repo()
    if _detect_secrets(content):
        return _json({"error": "Refusing to write likely secret/credential material."})
    full = _safe_path(file_path)
    rel = str(full.relative_to(REPO_PATH))
    if not _allowed_write_path(rel):
        return _json({"error": "Writes restricted to docs/, skills/, templates/, AGENTS.md, README.md, CONTRIBUTING.md", "path": rel})
    if mode not in {"replace", "append"}:
        return _json({"error": "mode must be replace or append"})
    # Append mode: reject content that would inject a second frontmatter block.
    # The downstream validator only inspects the first frontmatter block, so a
    # naive append of "---\n<keys>\n---\n..." silently corrupts the doc.
    if mode == "append":
        stripped = content.lstrip()
        if stripped.startswith("---\n") or stripped.startswith("---\r\n"):
            after_open = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            for line in after_open.splitlines():
                if line.rstrip() == "---":
                    return _json({
                        "error": "append rejected: content begins with a frontmatter block (would create duplicate frontmatter)",
                        "path": rel,
                        "rolled_back": True,
                    })

    # Serialize the full pull -> write -> validate -> commit -> push sequence
    # across every brain-mcp process (each MCP stdio client spawns its own),
    # so two concurrent updates can't both win the local commit and race the
    # remote push.
    with _interprocess_write_lock():
        _pull_ff()
        existed_before = full.exists()
        old = full.read_text(encoding="utf-8") if existed_before else ""
        full.parent.mkdir(parents=True, exist_ok=True)
        new = content.rstrip() + "\n" if mode == "replace" else old.rstrip() + "\n" + content.rstrip() + "\n"
        if _detect_secrets(new):
            return _json({"error": "Refusing to write because resulting file appears to contain secrets."})

        changed_docs = rel.startswith("docs/")
        paths = [rel] + (["docs/index.md"] if changed_docs else [])
        created = [rel] if not existed_before else []

        full.write_text(new, encoding="utf-8")
        try:
            validation = _validate_after_write(changed_docs=changed_docs)
            result = _commit_push(paths, commit_message)
        except Exception as exc:
            _restore_paths(paths, created)
            return _json({
                "error": f"update rolled back: {exc}",
                "path": rel,
                "rolled_back": True,
            })
    # Fire-and-forget consistency check on the just-written doc. Detached,
    # never blocks the response. Findings land in ~/.brein/consistency-queue.jsonl
    # and are pulled via brain_consistency_status.
    consistency_pid = None
    try:
        consistency_pid = consistency.spawn_detached(rel)
    except Exception:
        pass

    return _json({
        "path": rel,
        "validation": validation,
        "consistency_check_pid": consistency_pid,
        **result,
    })


@mcp.tool(name="brain_consistency_status")
@logged("brain_consistency_status")
def brain_consistency_status(max_results: int = 20, clear: bool = False) -> str:
    """Return pending consistency findings from background brain_update checks.

    Each brain_update spawns a detached worker that compares the new doc
    against its nearest semantic neighbors via an LLM judge. Findings can be:

      - auto_merge      near-duplicate of an existing doc (suggest merging)
      - contradiction   facts disagree with an existing doc
      - unresolved      potential conflict, judge unsure — worth user review

    Agents should call this periodically (e.g. once per session or after a
    burst of writes) and surface unresolved/contradiction findings to the
    user. `clear=True` empties the queue after returning the current set.
    """
    queue = consistency.read_queue()
    payload = {
        "queue_size": len(queue),
        "findings": [f.to_json() for f in queue[-max_results:]],
        "queue_path": str(consistency.QUEUE_PATH),
    }
    if clear:
        consistency.clear_queue()
        payload["cleared"] = True
    return _json(payload)


@mcp.tool(name="brain_audit")
@logged("brain_audit")
def brain_audit() -> str:
    """Summarize repo health for retrieval and write safety."""
    _ensure_repo()
    status = _run_git(["status", "--short"], check=True).stdout.strip()
    validate = _run_repo_cmd(
        ["python3", _bundled_script("validate_docs.py", "BRAIN_VALIDATE_SCRIPT")]
    ).stdout.strip()
    index = _run_repo_cmd(
        ["python3", _bundled_script("generate_index.py", "BRAIN_INDEX_SCRIPT"), "--check"]
    ).stdout.strip()
    docs = list(_iter_markdown("docs") or [])
    total = len(docs)
    with_fm = 0
    by_domain: dict[str, int] = {}
    for path in docs:
        text = path.read_text(encoding="utf-8")
        if _frontmatter(text):
            with_fm += 1
        parts = path.relative_to(REPO_PATH).parts
        if len(parts) > 1 and parts[0] == "docs" and "." not in parts[1]:
            by_domain[parts[1]] = by_domain.get(parts[1], 0) + 1
        elif len(parts) > 1 and parts[0] == "docs":
            by_domain["_top_level"] = by_domain.get("_top_level", 0) + 1
    log_lines = 0
    if LOG_PATH.exists():
        log_lines = sum(1 for _ in LOG_PATH.open("r", encoding="utf-8", errors="ignore"))
    vector_info: dict[str, Any] = {
        "index_path": str(VECTOR_INDEX_PATH),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "backend": _get_embedder_backend(),
        "exists": VECTOR_INDEX_PATH.exists(),
    }
    if VECTOR_INDEX_PATH.exists():
        try:
            cached = json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))
            vector_info.update({
                "built_at": cached.get("built_at"),
                "build_seconds": cached.get("build_seconds"),
                "entries": len(cached.get("entries", [])),
            })
        except Exception as exc:
            vector_info["error"] = f"unreadable index: {type(exc).__name__}"
    return _json({
        "repo": str(REPO_PATH),
        "clean": status == "",
        "dirty_status": status,
        "validate_docs": validate,
        "index_check": index,
        "docs_total": total,
        "docs_with_frontmatter": with_fm,
        "docs_by_domain": by_domain,
        "retrieval_log": str(LOG_PATH),
        "retrieval_log_lines": log_lines,
        "vector_index": vector_info,
    })


@mcp.tool(name="brain_retrieval_log")
@logged("brain_retrieval_log")
def brain_retrieval_log(question: str, hits: list[str] | None = None, used_docs: list[str] | None = None, outcome: str = "unknown") -> str:
    """Append retrieval telemetry. Search/read/answer auto-log; this is for manual outcome tagging."""
    _append_retrieval_log(question, hits, used_docs, outcome, kind="manual")
    return _json({"logged": True, "path": str(LOG_PATH)})


@mcp.tool(name="brain_eval_summary")
@logged("brain_eval_summary")
def brain_eval_summary(include_examples: bool = True) -> str:
    """Aggregate .brain/eval-log.jsonl into a readable summary.

    Splits stats by question_class and backend, filters out rows tagged
    `*-deprecated` from the headline rate (old methodology), and lists
    recent brain losses on internal_only/mixed questions — those are the
    rows worth patching in the brain.
    """
    import collections
    path = REPO_PATH / ".brain" / "eval-log.jsonl"
    if not path.exists():
        return "(no eval log yet)"
    rows = [json.loads(l) for l in path.open() if l.strip()]
    if not rows:
        return "(eval log empty)"

    robust = [r for r in rows if "deprecated" not in r.get("backend", "")]
    by_verdict = collections.Counter(r.get("verdict", "?") for r in robust)
    by_class = collections.Counter(r.get("question_class", "unknown") for r in robust)
    by_backend = collections.Counter(r.get("backend", "?") for r in robust)
    # Bucket triggers on the prefix before ":" so per-row rerun tags collapse.
    by_trigger = collections.Counter(
        (r.get("trigger", "?") or "?").split(":", 1)[0] for r in robust
    )

    def pct(n: int, d: int) -> str:
        return f"{round(100 * n / d, 1)}%" if d else "—"

    n = len(robust)
    brain_wins = by_verdict.get("brain_better", 0)
    ties = by_verdict.get("tie", 0)
    losses = by_verdict.get("no_brain_better", 0)

    # Brain losses on questions where brain *should* help (internal/mixed)
    loss_rows = [
        r for r in robust
        if r.get("verdict") == "no_brain_better"
        and r.get("question_class") in {"internal_only", "mixed"}
    ]

    # Cost / latency / token roll-ups, when the row has a `totals` block.
    def _totals(field: str) -> list[float]:
        out: list[float] = []
        for r in robust:
            t = r.get("totals") or {}
            v = t.get(field)
            if isinstance(v, (int, float)) and v > 0:
                out.append(float(v))
        return out

    costs = _totals("cost_usd")
    walls = _totals("wall_clock_ms")
    in_tokens = _totals("input_tokens")
    out_tokens = _totals("output_tokens")
    cache_tokens = _totals("cache_read_input_tokens")

    def _median(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[len(s) // 2]

    # Per-arm wall-clock medians, for "did brain save time" view.
    def _arm_lat(arm_key: str) -> float:
        xs = []
        for r in robust:
            m = r.get(arm_key) or {}
            v = m.get("latency_ms")
            if isinstance(v, (int, float)) and v > 0:
                xs.append(float(v))
        return _median(xs)

    brain_p50 = _arm_lat("brain_meta")
    no_brain_p50 = _arm_lat("no_brain_meta")

    # ── Per-query A/B comparison table (fast facts) ──────────────────────
    def _arm_field(arm_key: str, field: str) -> list[float]:
        xs = []
        for r in robust:
            m = r.get(arm_key) or {}
            v = m.get(field)
            if isinstance(v, (int, float)) and v >= 0:
                xs.append(float(v))
        return xs

    def _med(xs: list[float]) -> float:
        return _median(xs) if xs else 0.0

    def _mean(xs: list[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    def _total_tokens(meta: dict) -> float:
        return float(
            (meta.get("input_tokens") or 0)
            + (meta.get("output_tokens") or 0)
            + (meta.get("cache_read_input_tokens") or 0)
            + (meta.get("cache_creation_input_tokens") or 0)
        )

    rows_with_meta = [
        r for r in robust
        if isinstance(r.get("brain_meta"), dict) and isinstance(r.get("no_brain_meta"), dict)
    ]
    fast_facts_lines: list[str] = []
    if rows_with_meta:
        # Paired arrays — only keep rows where BOTH arms have the field present.
        def _paired(field: str, divisor: float = 1.0, require_truthy: bool = False) -> tuple[list[float], list[float]]:
            bx, nx = [], []
            for r in rows_with_meta:
                bv = (r["brain_meta"] or {}).get(field)
                nv = (r["no_brain_meta"] or {}).get(field)
                if not isinstance(bv, (int, float)) or not isinstance(nv, (int, float)):
                    continue
                if require_truthy and (not bv or not nv):
                    continue
                bx.append(bv / divisor)
                nx.append(nv / divisor)
            return bx, nx

        b_lat, n_lat = _paired("latency_ms", divisor=1000.0, require_truthy=True)
        b_out, n_out = _paired("output_tokens")
        b_cost, n_cost = _paired("cost_usd", require_truthy=True)

        # Token totals — paired across all four buckets.
        b_tot, n_tot = [], []
        for r in rows_with_meta:
            b_tot.append(_total_tokens(r["brain_meta"]))
            n_tot.append(_total_tokens(r["no_brain_meta"]))

        def _verb(brain_val: float, no_brain_val: float, unit: str, is_cost: bool = False) -> str:
            delta = no_brain_val - brain_val
            if abs(delta) < 0.01:
                return f"roughly equal ({unit}{abs(delta):.1f})"
            if is_cost:
                if delta > 0:
                    return f"**brain saves {unit}{delta:.2f}** per query"
                return f"brain costs ~{unit}{-delta:.2f} MORE per query"
            if delta > 0:
                return f"**brain saves {delta:.1f}{unit}**"
            return f"brain uses {-delta:.1f}{unit} MORE"

        def _fmt_tok(n: float) -> str:
            return f"{n/1000:.0f}k" if n >= 1000 else f"{n:.0f}"

        bl_med, nl_med = _med(b_lat), _med(n_lat)
        bt_med, nt_med = _med(b_tot), _med(n_tot)
        bo_med, no_med = _med(b_out), _med(n_out)
        bc_med, nc_med = _med(b_cost), _med(n_cost)

        # Paired deltas — the *real* "per-query saving" stats.
        lat_deltas = [n_lat[i] - b_lat[i] for i in range(len(b_lat))]
        tot_deltas = [n_tot[i] - b_tot[i] for i in range(len(b_tot))]
        cost_deltas = [n_cost[i] - b_cost[i] for i in range(len(b_cost))]

        lat_med_save = _med(lat_deltas)
        lat_mean_save = _mean(lat_deltas)
        tot_med_save = _med(tot_deltas)
        tot_mean_save = _mean(tot_deltas)
        cost_med_save = _med(cost_deltas)

        output_shorter_pct = (1 - bo_med / no_med) * 100 if no_med else 0
        out_delta_str = (
            f"**brain answer is {output_shorter_pct:.0f}% shorter**"
            if output_shorter_pct > 0
            else f"brain answer is {-output_shorter_pct:.0f}% longer"
        )

        def _delta_lat(save_med: float, save_mean: float) -> str:
            if save_med > 0:
                return f"**brain saves {save_med:.1f}s** (mean {save_mean:.1f}s)"
            return f"brain uses {-save_med:.1f}s MORE (mean {-save_mean:.1f}s)"

        def _delta_tok(save_med: float, save_mean: float) -> str:
            if save_med > 0:
                return f"**brain saves {save_med/1000:.0f}k tokens** (mean {save_mean/1000:.0f}k)"
            return f"brain uses {-save_med/1000:.0f}k MORE (mean {-save_mean/1000:.0f}k)"

        def _delta_cost(save_med: float) -> str:
            if save_med >= 0.01:
                return f"**brain saves ${save_med:.2f}** per query"
            if save_med <= -0.01:
                return f"brain costs ~${-save_med:.2f} MORE per query"
            return "roughly equal"

        fast_facts_lines = [
            "",
            f"## Fast facts ({len(rows_with_meta)} A/B rows, paired per-query deltas)",
            "",
            f"| Per query              | Brain arm  | No-brain arm | Delta                                    |",
            f"|------------------------|------------|--------------|------------------------------------------|",
            f"| Wall-clock (median)    | {bl_med:>6.1f}s   | {nl_med:>6.1f}s     | {_delta_lat(lat_med_save, lat_mean_save)}",
            f"| Tokens consumed (med)  | {_fmt_tok(bt_med):>8}   | {_fmt_tok(nt_med):>10}   | {_delta_tok(tot_med_save, tot_mean_save)}",
            f"| Output tokens (med)    | {bo_med:>8.0f}   | {no_med:>10.0f}   | {out_delta_str}",
        ]
        if bc_med and nc_med:
            fast_facts_lines.append(
                f"| API-equiv cost (med)   | ${bc_med:>7.2f}   | ${nc_med:>9.2f}   | {_delta_cost(cost_med_save)}"
            )
        fast_facts_lines.append("")

    lines = [
        "# Brain eval summary",
        f"Total rows: {len(rows)}  (robust: {n}, deprecated: {len(rows) - n})",
        "",
        "## Headline rates (robust subset)",
        f"  brain_better:    {brain_wins}  ({pct(brain_wins, n)})",
        f"  tie:             {ties}  ({pct(ties, n)})",
        f"  no_brain_better: {losses}  ({pct(losses, n)})",
        *fast_facts_lines,
        "## Cost / latency / tokens (rows that captured `totals`)",
    ]
    if len(costs) >= 2:
        lines += [
            f"  rows with cost data:    {len(costs)}",
            f"  total cost spent:       ${sum(costs):.4f}",
            f"  median row cost:        ${_median(costs):.4f}",
            f"  total wall-clock:       {sum(walls)/1000:.1f}s",
            f"  median wall-clock/row:  {_median(walls)/1000:.1f}s",
            f"  total input tokens:     {int(sum(in_tokens)):,}",
            f"  total output tokens:    {int(sum(out_tokens)):,}",
            f"  total cache-read tok:   {int(sum(cache_tokens)):,}",
            "",
            "## Per-arm median wall-clock",
            f"  brain arm:    {brain_p50/1000:.1f}s",
            f"  no-brain arm: {no_brain_p50/1000:.1f}s",
            f"  delta:        {(no_brain_p50 - brain_p50)/1000:+.1f}s  (positive = brain faster)",
            "",
        ]
    else:
        lines += [
            f"  (only {len(costs)} row(s) with cost data — continuous-loop "
            f"`cli:claude+tools` rows populate this, accumulating over time)",
            "",
        ]
    lines += [
        "## By question_class",
        *[f"  {k}: {v}" for k, v in by_class.most_common()],
        "",
        "## By backend",
        *[f"  {k}: {v}" for k, v in by_backend.most_common()],
        "",
        "## By trigger",
        *[f"  {k}: {v}" for k, v in by_trigger.most_common()],
        "",
        f"## Brain losses on internal/mixed ({len(loss_rows)}) — candidates to patch",
    ]
    if include_examples:
        for r in loss_rows[-15:]:
            lines.append(f"  - {r.get('question', '')[:90]}")
            lines.append(f"    why: {r.get('reason', '')[:140]}")
    return "\n".join(lines)


def _startup_warmup() -> None:
    """Eagerly load fastembed + vector index in a background thread so the
    stdio handshake isn't blocked but the first user search doesn't pay the
    cold-start cliff (~5–15s on bge-small)."""
    health = _vector_health()
    if health["degraded"]:
        print(
            f"[brain-mcp] WARNING: {health['warning']} (backend={health['backend']})",
            file=sys.stderr,
            flush=True,
        )

    def _bg_warmup() -> None:
        import time as _t
        try:
            t0 = _t.time()
            backend = _get_embedder_backend()  # forces fastembed model load
            print(
                f"[brain-mcp] embedder warm: {backend} in {round(_t.time()-t0, 2)}s",
                file=sys.stderr,
                flush=True,
            )
            t0 = _t.time()
            idx = _load_vector_index(directory="docs")  # parses + caches index
            print(
                f"[brain-mcp] vector index warm: {len(idx.get('entries', []))} chunks "
                f"in {round(_t.time()-t0, 2)}s (built_at={idx.get('built_at')})",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"[brain-mcp] warmup failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    threading.Thread(target=_bg_warmup, name="cb-warmup", daemon=True).start()


def main() -> None:
    if os.environ.get("BRAIN_MCP_SELF_TEST") == "1":
        print(brain_audit())
        return
    if os.environ.get("BRAIN_EVAL_DISABLE") == "1":
        # No-brain baseline subprocess in continuous eval — exit cleanly so
        # the child claude session runs without any brain tools.
        print("[brain-mcp] BRAIN_EVAL_DISABLE=1 — exiting (no-brain baseline).", file=sys.stderr, flush=True)
        return
    try:
        _ensure_repo()
        _startup_warmup()
    except Exception as exc:
        print(f"[brain-mcp] startup error: {exc}", file=sys.stderr, flush=True)

    # Transport: stdio (default, one server per client) or streamable-http
    # (one daemon, many clients — model loaded once). Set
    # BRAIN_MCP_TRANSPORT=http to share one process across N MCP clients.
    transport = os.environ.get("BRAIN_MCP_TRANSPORT", "stdio").lower()
    if transport in ("http", "streamable-http", "sse"):
        host = os.environ.get("BRAIN_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("BRAIN_MCP_PORT", "8765"))
        mcp.settings.host = host
        mcp.settings.port = port
        print(
            f"[brain-mcp] listening on http://{host}:{port}/mcp (streamable-http)",
            file=sys.stderr,
            flush=True,
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
