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
from .rerank import _maybe_rerank
from .telemetry import logged
from .shared import (
    _allowed_write_path,
    _append_retrieval_log,
    _detect_secrets,
    _ensure_repo,
    _frontmatter,
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
    if changed_docs:
        _run_repo_cmd(["python3", index_script], check=True)
        messages.append("regenerated docs/index.md")
    _run_repo_cmd(["python3", "scripts/validate_docs.py"], check=True)
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
    # ponytail: push in background — caller doesn't wait the ~1s network round-trip.
    # Failure logged to stderr; next op's _pull_ff catches divergence.
    threading.Thread(target=_bg_push, name="cb-push", daemon=True).start()
    return {"changed": True, "pushed": "pending", "clean": status == "", "dirty_status": status, "head": head}


@mcp.tool(name="brain_list")
@logged("brain_list")
def brain_list(directory: str = "docs", max_results: int = 500) -> str:
    """List markdown files in the brain."""
    _ensure_repo()
    files = []
    for path in _iter_markdown(directory) or []:
        files.append(str(path.relative_to(REPO_PATH)))
        if len(files) >= max_results:
            break
    return _json({"repo": str(REPO_PATH), "files": files, "truncated": len(files) >= max_results})


@mcp.tool(name="brain_search")
@logged("brain_search")
def brain_search(
    query: str,
    directory: str = "docs",
    domain: str | None = None,
    tag: str | None = None,
    status: str | None = None,
    max_results: int = 10,
    mode: str = "hybrid",
    rerank: bool = False,
    rerank_top_k: int = 25,
    rerank_method: str = "llm",
    force_rebuild_vector_index: bool = False,
) -> str:
    """Hybrid brain search: keyword/BM25-ish plus vector semantic retrieval.

    mode can be `hybrid`, `keyword`, or `vector`. Hybrid preserves exact-match
    strengths while adding semantic recall for questions whose wording differs
    from the docs.
    """
    _ensure_repo()
    if not query.strip():
        return _json({"error": "query is required"})
    if mode not in {"hybrid", "keyword", "vector"}:
        return _json({"error": "mode must be hybrid, keyword, or vector"})
    if rerank_method not in {"llm", "heuristic"}:
        rerank_method = "llm"

    toks = _tokens(query) or [query.lower()]
    keyword_results: list[dict[str, Any]] = []
    for path in _iter_markdown(directory) or []:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        fm = _frontmatter(text)
        if not _matches_filters(path, fm, domain, status, tag):
            continue
        score = _score(path, text, fm, toks)
        if score <= 0:
            continue
        keyword_results.append({
            "path": str(path.relative_to(REPO_PATH)),
            "keyword_score": float(score),
            "title": fm.get("title", path.stem),
            "status": fm.get("status"),
            "tags": fm.get("tags", []),
            "source_of_truth": fm.get("source_of_truth"),
            "snippets": _line_snippets(text, toks),
        })
    keyword_results.sort(key=lambda r: r["keyword_score"], reverse=True)

    vector_results: list[dict[str, Any]] = []
    vector_meta: dict[str, Any] = {"enabled": False}
    if mode in {"hybrid", "vector"}:
        vector_results, vector_meta = _best_vector_hits(
            query=query,
            directory=directory,
            domain=domain,
            tag=tag,
            status=status,
            limit=max(max_results * 4, min(rerank_top_k if rerank else 25, RERANK_MAX_TOP_K), 25),
            force_rebuild=force_rebuild_vector_index,
        )
        vector_meta["enabled"] = True

    if mode == "keyword":
        results = []
        for r in keyword_results:
            results.append({**r, "score": round(r["keyword_score"], 3), "retrieval": "keyword"})
        results, rerank_meta = _maybe_rerank(query, toks, results, rerank, rerank_method, rerank_top_k)
        top = results[:max_results]
        _append_retrieval_log(query, [r["path"] for r in top], None, "search_keyword", kind="search", extra={"mode": mode})
        return _json({"query": query, "tokens": toks, "mode": mode, "rerank": rerank_meta, "results": top, "truncated": len(results) > max_results})

    if mode == "vector":
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
        _append_retrieval_log(query, [r["path"] for r in top], None, "search_vector", kind="search", extra={"mode": mode, "backend": vector_meta.get("backend")})
        return _json({"query": query, "tokens": toks, "mode": mode, "vector": vector_meta, "rerank": rerank_meta, "results": top, "truncated": len(results) > max_results})

    # Hybrid fusion. Normalize keyword and vector scores independently before
    # combining so one scale cannot dominate the other.
    max_kw = max([r["keyword_score"] for r in keyword_results], default=0.0) or 1.0
    max_vec = max([r["vector_score"] for r in vector_results], default=0.0) or 1.0
    fused: dict[str, dict[str, Any]] = {}
    for r in keyword_results:
        item = fused.setdefault(r["path"], {**r, "keyword_score": 0.0, "vector_score": 0.0, "retrieval": []})
        item.update({k: v for k, v in r.items() if k not in {"keyword_score", "snippets"}})
        item["keyword_score"] = r["keyword_score"]
        item["snippets"] = r.get("snippets", [])
        item["retrieval"].append("keyword")
    for r in vector_results:
        item = fused.setdefault(r["path"], {
            "path": r["path"],
            "title": r.get("title"),
            "status": r.get("status"),
            "tags": r.get("tags", []),
            "source_of_truth": r.get("source_of_truth"),
            "keyword_score": 0.0,
            "vector_score": 0.0,
            "snippets": [],
            "retrieval": [],
        })
        item["vector_score"] = r["vector_score"]
        if r.get("vector_snippet") and not item.get("snippets"):
            item["snippets"] = [r["vector_snippet"]]
        item["retrieval"].append("vector")
    results = []
    for item in fused.values():
        kw_norm = item["keyword_score"] / max_kw if item["keyword_score"] > 0 else 0.0
        vec_norm = item["vector_score"] / max_vec if item["vector_score"] > 0 else 0.0
        fused_score = HYBRID_KEYWORD_WEIGHT * kw_norm + HYBRID_VECTOR_WEIGHT * vec_norm
        if str(item.get("source_of_truth", "")).lower() == "true":
            fused_score += 0.02
        results.append({
            "path": item["path"],
            "score": round(fused_score, 6),
            "keyword_score": round(item["keyword_score"], 3),
            "vector_score": round(item["vector_score"], 6),
            "title": item.get("title"),
            "status": item.get("status"),
            "tags": item.get("tags", []),
            "source_of_truth": item.get("source_of_truth"),
            "snippets": item.get("snippets", []),
            "retrieval": sorted(set(item.get("retrieval", []))),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    results, rerank_meta = _maybe_rerank(query, toks, results, rerank, rerank_method, rerank_top_k)
    top = results[:max_results]
    _append_retrieval_log(query, [r["path"] for r in top], None, "search_hybrid", kind="search", extra={"mode": mode, "backend": vector_meta.get("backend")})
    return _json({
        "query": query,
        "tokens": toks,
        "mode": mode,
        "weights": {"keyword": HYBRID_KEYWORD_WEIGHT, "vector": HYBRID_VECTOR_WEIGHT},
        "vector": vector_meta,
        "rerank": rerank_meta,
        "results": top,
        "truncated": len(results) > max_results,
    })


@mcp.tool(name="brain_read")
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


@mcp.tool(name="brain_answer")
@logged("brain_answer")
def brain_answer(question: str, max_docs: int = 5) -> str:
    """Return an evidence bundle for a question: ranked docs + snippets + citations.

This is intentionally retrieval-grounded, not a hallucinated final answer.
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
    return _json({"path": rel, "validation": validation, **result})


@mcp.tool(name="brain_audit")
@logged("brain_audit")
def brain_audit() -> str:
    """Summarize repo health for retrieval and write safety."""
    _ensure_repo()
    status = _run_git(["status", "--short"], check=True).stdout.strip()
    validate = _run_repo_cmd(["python3", "scripts/validate_docs.py"]).stdout.strip()
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


@mcp.tool(name="brain_classify")
@logged("brain_classify")
def brain_classify(text: str) -> str:
    """Heuristically route text to brain, personal brain, memory, nowhere, or split."""
    lower = text.lower()
    if _detect_secrets(text):
        return _json({"destination": "nowhere", "reason": "Looks like secret/credential material."})
    work_terms = ["customer", "client", "investor", "team", "decision", "deadline", "company", "project", "meeting", "incident", "postmortem"]
    personal_terms = ["family", "health", "home", "travel", "diet", "personal", "private"]
    if any(k in lower for k in work_terms) and any(k in lower for k in personal_terms):
        dest = "split"
        reason = "Mixed work + personal signal. Save only team-shareable parts to the brain."
    elif any(k in lower for k in work_terms):
        dest = "brain"
        reason = "Looks like durable work/team knowledge."
    elif any(k in lower for k in ["prefer", "timezone", "style", "always"]):
        dest = "assistant_memory"
        reason = "Looks like a durable user preference/profile fact — for assistant memory, not the shared brain."
    else:
        dest = "ignore"
        reason = "No clear durable signal — likely ephemeral chat."
    return _json({"destination": dest, "reason": reason})


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
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
