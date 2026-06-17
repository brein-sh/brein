#!/usr/bin/env python3
"""Policy-aware MCP server for the brain.

Exposes curated retrieval + safe write tools over stdio. Designed for Hermes,
Claude Code, Codex, or any MCP client.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

def _p(s: str) -> Path:
    return Path(s).expanduser().resolve()

REPO_PATH = _p(os.environ.get("BRAIN_REPO", "~/.braincorp/brain"))
DOCS_PATH = REPO_PATH / "docs"
MAX_READ_CHARS = int(os.environ.get("BRAIN_MAX_READ_CHARS", "80000"))
LOG_PATH = _p(os.environ.get("BRAIN_RETRIEVAL_LOG", "~/.braincorp/retrieval-log.jsonl"))
VECTOR_INDEX_PATH = _p(os.environ.get("BRAIN_VECTOR_INDEX", "~/.braincorp/vector-index.json"))
EMBEDDING_MODEL_NAME = os.environ.get("BRAIN_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_CHUNK_CHARS = int(os.environ.get("BRAIN_VECTOR_CHUNK_CHARS", "1400"))
VECTOR_CHUNK_OVERLAP = int(os.environ.get("BRAIN_VECTOR_CHUNK_OVERLAP", "250"))
HYBRID_KEYWORD_WEIGHT = float(os.environ.get("BRAIN_HYBRID_KEYWORD_WEIGHT", "0.55"))
HYBRID_VECTOR_WEIGHT = float(os.environ.get("BRAIN_HYBRID_VECTOR_WEIGHT", "0.45"))
HASH_EMBED_DIMS = int(os.environ.get("BRAIN_HASH_EMBED_DIMS", "384"))
RERANK_MAX_TOP_K = int(os.environ.get("BRAIN_RERANK_MAX_TOP_K", "25"))
RERANK_TIMEOUT_SECONDS = float(os.environ.get("BRAIN_RERANK_TIMEOUT_SECONDS", "25"))
RERANK_SNIPPET_CHARS = int(os.environ.get("BRAIN_RERANK_SNIPPET_CHARS", "240"))
RERANK_SNIPPET_COUNT = int(os.environ.get("BRAIN_RERANK_SNIPPET_COUNT", "2"))
RERANK_PROVIDER_DEFAULT = os.environ.get("BRAIN_RERANK_PROVIDER", "openai-codex")
RERANK_MODEL_DEFAULT = os.environ.get(
    "BRAIN_RERANK_MODEL",
    "gpt-5.4-mini" if RERANK_PROVIDER_DEFAULT == "openai-codex" else "",
)
RERANK_COMMAND_DEFAULT = os.environ.get("BRAIN_RERANK_COMMAND", "")
RERANK_BIN_DEFAULT = os.environ.get("BRAIN_RERANK_BIN", "")

_EMBEDDER = None
_EMBEDDER_BACKEND = "uninitialized"

mcp = FastMCP(
    "Brain",
    instructions=(
        "This is the user's brain — a git-backed markdown repo of their durable knowledge. "
        "USE these tools, do NOT fall back to ls/Read/Write/grep, even if the brain looks small. "
        "brain_list: enumerate files. "
        "brain_read: read a file by path. "
        "brain_search: hybrid keyword+vector retrieval — always prefer this over grep. "
        "brain_write: write or update a file (commits + pushes). "
        "Never store secrets. Search before writing."
    ),
)

SECRET_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd|api[_-]?key|x-api-key|secret|token|auth[_-]?token|private[_-]?key)\s*[:=]\s*[^\s`'\"]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"(?i)seed phrase\s*[:=]"),
]

# ponytail: permissive by default — brain is the user's. The path-traversal
# guard in _resolve_write_path is what actually keeps us safe.
ALLOWED_WRITE_PREFIXES = tuple(
    p.strip() for p in os.environ.get("BRAIN_WRITE_PREFIXES", "").split(",") if p.strip()
) or ("",)  # "" matches every relative path
ALLOWED_ROOT_WRITES = {"AGENTS.md", "README.md", "CONTRIBUTING.md", "CLAUDE.md"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "our", "the", "to", "what", "when",
    "where", "who", "why", "with", "does", "do", "we", "have", "has",
    "against", "about",
}


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
    # "" or "." means repo root — used by list/search to scan the whole brain
    if not file_path or file_path.strip() == ".":
        return REPO_PATH
    if file_path.strip() == "..":
        raise ValueError("file_path escapes repo")
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


_EXCLUDED_DIRS = {".git", "node_modules", ".venv", ".next", "__pycache__", "dist", "build"}


def _iter_markdown(directory: str = "."):
    base = _safe_path(directory)
    if not base.exists():
        return
    paths = base.rglob("*.md") if base.is_dir() else [base]
    for path in paths:
        if not path.is_file():
            continue
        parts = path.relative_to(REPO_PATH).parts
        if any(p in _EXCLUDED_DIRS for p in parts):
            continue
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
    # Prefer curated source-of-truth docs a bit.
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


def _normalize_rerank_method(method: str | None) -> str:
    method_l = (method or "llm").strip().lower()
    if method_l in {"", "auto"}:
        return "llm"
    return method_l


def _normalize_rerank_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        return []
    return [str(tag) for tag in tags if str(tag).strip()]


def _normalize_rerank_retrieval(retrieval: Any) -> list[str]:
    if isinstance(retrieval, str):
        retrieval = [retrieval]
    if not isinstance(retrieval, list):
        return []
    return sorted({str(item) for item in retrieval if str(item).strip()})


def _compact_rerank_candidate(result: dict[str, Any]) -> dict[str, Any]:
    snippets: list[dict[str, Any]] = []
    for snippet in result.get("snippets", [])[:RERANK_SNIPPET_COUNT]:
        if not isinstance(snippet, dict):
            continue
        snippet_text = _truncate_text(str(snippet.get("snippet", "")), RERANK_SNIPPET_CHARS)
        if not snippet_text:
            continue
        snippet_item: dict[str, Any] = {"snippet": snippet_text}
        if snippet.get("line") is not None:
            snippet_item["line"] = snippet.get("line")
        snippets.append(snippet_item)
    return {
        "path": str(result.get("path", "")),
        "title": str(result.get("title", "")),
        "tags": _normalize_rerank_tags(result.get("tags", [])),
        "source_of_truth": _truthy(result.get("source_of_truth")),
        "retrieval": _normalize_rerank_retrieval(result.get("retrieval", [])),
        "score": round(float(result.get("score", 0.0) or 0.0), 6),
        "snippets": snippets,
    }


def _build_rerank_prompt(query: str, candidates: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {
            "query": query,
            "candidates": [_compact_rerank_candidate(candidate) for candidate in candidates],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "You are a strict document reranker for the brain.\n"
        "Return JSON only. No markdown, no prose, no code fences.\n\n"
        "Task: score each candidate for how directly it helps answer the query.\n"
        "Do not answer the query. Do not invent new facts.\n"
        "Prefer specific source-of-truth docs over index pages, maps, generic architecture docs, or broad summaries unless the query explicitly asks for navigation or overview.\n"
        "For defensibility / moat / competitive / pricing / regulatory / customer-risk / venue-copying questions, be strict: only direct evidence should score high; broad or speculative docs should score low.\n"
        "Use the candidate's own path/title/tags/source_of_truth/retrieval/score/snippets only.\n"
        "Score scale: 0 = irrelevant, 1 = weakly related, 2 = somewhat related, 3 = relevant, 4 = highly relevant, 5 = directly answers or is the canonical source.\n"
        "Return exact JSON in the shape {\"scores\":[{\"path\":\"...\",\"score\":0-5,\"reason\":\"short\"}]}.\n"
        "Include exactly one entry for each input candidate path. Keep reason short (under ~12 words).\n\n"
        f"INPUT_JSON={payload}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(raw[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a JSON object")


def _rerank_reason_from_signals(signals: dict[str, Any]) -> str:
    priority = [
        ("source_of_truth", "source of truth"),
        ("keyword_and_vector", "keyword+vector"),
        ("all_query_tokens", "all tokens"),
        ("most_query_tokens", "most tokens"),
        ("phrase_in_title", "phrase in title"),
        ("phrase_match", "phrase match"),
        ("field_match", "field match"),
        ("token_coverage", "query coverage"),
        ("snippet_density", "snippet density"),
        ("demoted_index", "index demoted"),
        ("demoted_broad_doc", "broad doc demoted"),
    ]
    phrases = [label for key, label in priority if signals.get(key)]
    if not phrases:
        return "baseline relevance"
    return "; ".join(phrases[:3])


def _is_indexish_path(path: str) -> bool:
    return path.endswith("/index.md") or path == "docs/index.md"


def _is_broad_reference(path: str, title: str) -> bool:
    hay = f"{path} {title}".lower()
    return any(marker in hay for marker in ("repo-map", "architecture", "overview", "index", "map"))


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        variants.add(stem)
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            variants.add(stem[:-1])
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return {v for v in variants if len(v) >= 2}


def _contains_variant(haystack: str, token: str) -> bool:
    return any(v in haystack for v in _token_variants(token))


def _variant_count(haystack: str, token: str) -> int:
    return max((haystack.count(v) for v in _token_variants(token)), default=0)


def _rerank_heuristic(query: str, toks: list[str], results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Deterministic local quality-control reranker for already-retrieved candidates.

    This is intentionally lightweight (no external API/model calls). It preserves
    retrieval recall while nudging candidates using exact query/result alignment,
    source authority, and broad-doc demotions.
    """
    if not results:
        return results

    query_l = query.strip().lower()
    nav_terms = {"index", "map", "navigation", "navigate", "overview", "toc", "table", "contents"}
    asks_for_nav = any(t in nav_terms for t in toks) or any(term in query_l for term in nav_terms)
    base_scores = [float(r.get("score", 0.0) or 0.0) for r in results]
    max_base = max(base_scores) or 1.0
    has_specific = any(not _is_broad_reference(str(r.get("path", "")), str(r.get("title", ""))) for r in results)
    reranked: list[dict[str, Any]] = []

    for original_rank, result in enumerate(results, start=1):
        parts = _result_text(result)
        combined = " ".join(parts.values())
        snippet = parts["snippets"]
        token_count = max(len(toks), 1)

        title_hits = sum(1 for t in toks if _contains_variant(parts["title"], t))
        tag_hits = sum(1 for t in toks if _contains_variant(parts["tags"], t))
        path_hits = sum(1 for t in toks if _contains_variant(parts["path"], t))
        combined_hits = sum(1 for t in toks if _contains_variant(combined, t))
        coverage = combined_hits / token_count

        # Shorter snippets with many hits are usually more answer-focused than
        # broad files whose snippets mention a term incidentally.
        snippet_words = max(len(re.findall(r"[A-Za-z0-9_.@-]{2,}", snippet)), 1)
        snippet_density = min(sum(_variant_count(snippet, t) for t in toks) / snippet_words, 0.20) / 0.20

        score = 0.0
        signals: dict[str, Any] = {}
        base_norm = float(result.get("score", 0.0) or 0.0) / max_base
        score += 0.35 * base_norm
        signals["base_norm"] = round(base_norm, 4)

        score += 0.22 * coverage
        signals["token_coverage"] = round(coverage, 4)
        if coverage >= 1.0:
            score += 0.08
            signals["all_query_tokens"] = True
        elif coverage >= 0.75:
            score += 0.04
            signals["most_query_tokens"] = True

        field_score = (0.10 * min(title_hits / token_count, 1.0)) + (0.05 * min(tag_hits / token_count, 1.0)) + (0.04 * min(path_hits / token_count, 1.0))
        score += field_score
        if field_score:
            signals["field_match"] = round(field_score, 4)

        score += 0.08 * snippet_density
        signals["snippet_density"] = round(snippet_density, 4)

        if query_l and query_l in parts["title"]:
            score += 0.15
            signals["phrase_in_title"] = True
        elif query_l and query_l in combined:
            score += 0.06
            signals["phrase_match"] = True

        retrieval = result.get("retrieval", [])
        if isinstance(retrieval, str):
            retrieval_set = {retrieval}
        else:
            retrieval_set = set(retrieval)
        if {"keyword", "vector"}.issubset(retrieval_set):
            score += 0.08
            signals["keyword_and_vector"] = True

        if _truthy(result.get("source_of_truth")):
            score += 0.06
            signals["source_of_truth"] = True

        path = str(result.get("path", ""))
        is_indexish = _is_indexish_path(path)
        is_broad = _is_broad_reference(path, str(result.get("title", "")))
        if is_indexish and not asks_for_nav:
            score -= 0.12
            signals["demoted_index"] = True
        if has_specific and is_broad and not asks_for_nav:
            score -= 0.08
            signals["demoted_broad_doc"] = True

        # Stable tie-breaker: keep a tiny preference for the original fusion order.
        score += max(0.0, 0.0001 * (len(results) - original_rank))
        reranked.append({
            **result,
            "rerank_score": round(score, 6),
            "rerank_reason": _rerank_reason_from_signals(signals),
            "rerank_signals": signals,
        })

    reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:top_k] + reranked[top_k:]


def _rerank_sort_key(result: dict[str, Any], original_rank: int) -> tuple[float, float, int]:
    return (
        float(result.get("rerank_score", result.get("score", 0.0)) or 0.0),
        float(result.get("score", 0.0) or 0.0),
        -original_rank,
    )


def _rerank_llm_command(provider: str | None, model: str | None) -> tuple[list[str] | None, dict[str, Any]]:
    if RERANK_COMMAND_DEFAULT.strip():
        cmd = shlex.split(RERANK_COMMAND_DEFAULT)
        return cmd if cmd else None, {
            "provider": provider,
            "model": model,
            "command": cmd[0] if cmd else None,
            "command_source": "env:BRAIN_RERANK_COMMAND",
        }

    hermes_bin = RERANK_BIN_DEFAULT.strip() or shutil.which("hermes") or "/opt/hermes/.venv/bin/hermes"
    if not hermes_bin or not Path(hermes_bin).exists():
        return None, {"provider": provider, "model": model, "error": "hermes binary not found"}

    cmd = [hermes_bin, "--ignore-user-config"]
    if provider:
        cmd.extend(["--provider", provider])
    if model:
        cmd.extend(["-m", model])
    cmd.append("-z")
    return cmd, {
        "provider": provider,
        "model": model,
        "command": hermes_bin,
        "command_source": "hermes-cli",
        "ignore_user_config": True,
    }


def _rerank_llm(query: str, candidates: list[dict[str, Any]], provider: str | None, model: str | None, timeout_seconds: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        return candidates, {"provider": provider, "model": model, "command_source": None}

    command, command_meta = _rerank_llm_command(provider, model)
    if not command:
        raise RuntimeError(command_meta.get("error") or "no rerank command available")

    prompt = _build_rerank_prompt(query, candidates)
    started = time.time()
    try:
        completed = subprocess.run(
            [*command, prompt],
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"rerank timed out after {timeout_seconds}s") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        raise RuntimeError(stderr or stdout or f"rerank command exited {completed.returncode}")
    payload = _extract_json_object(stdout)
    scores = payload.get("scores")
    if not isinstance(scores, list):
        raise RuntimeError("rerank response missing scores array")

    by_path = {str(candidate.get("path", "")): candidate for candidate in candidates}
    parsed: dict[str, dict[str, Any]] = {}
    for item in scores:
        if not isinstance(item, dict):
            raise RuntimeError("rerank response contained a non-object score entry")
        path = str(item.get("path", "")).strip()
        if path not in by_path:
            raise RuntimeError(f"rerank response included unexpected path: {path}")
        raw_score = item.get("score", 0)
        try:
            score = float(raw_score)
        except Exception as exc:
            raise RuntimeError(f"rerank score for {path} was not numeric") from exc
        if score < 0 or score > 5:
            raise RuntimeError(f"rerank score for {path} out of range: {score}")
        parsed[path] = {
            "score": round(score, 3),
            "reason": _truncate_text(str(item.get("reason", "")), 120),
        }
    if len(parsed) != len(candidates):
        missing = sorted(set(by_path) - set(parsed))
        raise RuntimeError(f"rerank response missing paths: {', '.join(missing)}")

    reranked: list[dict[str, Any]] = []
    for original_rank, result in enumerate(candidates, start=1):
        path = str(result.get("path", ""))
        entry = parsed[path]
        reranked.append({
            **result,
            "rerank_score": entry["score"],
            "rerank_reason": entry["reason"],
        })
    reranked.sort(key=lambda r: _rerank_sort_key(r, candidates.index(by_path[str(r.get("path", ""))]) + 1), reverse=True)
    command_meta.update({
        "elapsed_seconds": round(time.time() - started, 3),
        "response_chars": len(stdout),
        "model": model,
        "provider": provider,
    })
    return reranked, command_meta


def _maybe_rerank(query: str, toks: list[str], results: list[dict[str, Any]], enabled: bool, method: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_method = _normalize_rerank_method(method)
    top_k = max(1, min(int(top_k or 1), RERANK_MAX_TOP_K))
    meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "requested_method": requested_method,
        "method": requested_method if enabled else requested_method,
        "fallback_used": False,
        "top_k": min(top_k, len(results)) if enabled else 0,
        "provider": None,
        "model": None,
        "command": None,
    }
    if not enabled or not results:
        return results, meta

    candidates = results[:top_k]
    tail = results[top_k:]
    tail_annotated = [
        {**item, "rerank_score": round(float(item.get("score", 0.0) or 0.0), 6), "rerank_reason": "outside rerank top_k"}
        for item in tail
    ]
    if requested_method == "heuristic":
        reranked = _rerank_heuristic(query, toks, candidates, top_k)
        meta["method"] = "heuristic"
        return reranked + tail_annotated, meta

    if requested_method not in {"llm", "heuristic"}:
        meta["warning"] = f"unknown rerank_method '{method}', used llm"

    provider = RERANK_PROVIDER_DEFAULT.strip() or None
    model = RERANK_MODEL_DEFAULT.strip() or None
    try:
        reranked, llm_meta = _rerank_llm(query, candidates, provider, model, RERANK_TIMEOUT_SECONDS)
        meta.update(llm_meta)
        meta["method"] = "llm"
        return reranked + tail_annotated, meta
    except Exception as exc:
        meta["fallback_used"] = True
        meta["method"] = "heuristic"
        meta["error"] = str(exc)
        reranked = _rerank_heuristic(query, toks, candidates, top_k)
        return reranked + tail_annotated, meta


def _norm_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [float(v) / norm for v in values]


def _hash_embedding(text: str, dims: int = HASH_EMBED_DIMS) -> list[float]:
    """Deterministic local fallback embedding.

    Weaker than neural embeddings, but keeps vector retrieval available if the
    optional fastembed backend or model download is unavailable.
    """
    vec = [0.0] * dims
    lower = text.lower()
    terms = re.findall(r"[A-Za-z0-9_.@-]{2,}", lower)
    grams = [lower[i:i + 4] for i in range(max(0, len(lower) - 3)) if not lower[i:i + 4].isspace()]
    for term in terms + grams[:4000]:
        digest = hashlib.blake2b(term.encode("utf-8", errors="ignore"), digest_size=8).digest()
        h = int.from_bytes(digest, "big", signed=False)
        vec[h % dims] += 1.0 if h & 1 == 0 else -1.0
    return _norm_vector(vec)


def _get_embedder_backend() -> str:
    global _EMBEDDER, _EMBEDDER_BACKEND
    if _EMBEDDER_BACKEND != "uninitialized":
        return _EMBEDDER_BACKEND
    try:
        from fastembed import TextEmbedding  # type: ignore

        _EMBEDDER = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        _EMBEDDER_BACKEND = f"fastembed:{EMBEDDING_MODEL_NAME}"
    except Exception as exc:  # optional dependency/model path can fail in offline envs
        _EMBEDDER = None
        _EMBEDDER_BACKEND = f"hash-fallback:{type(exc).__name__}"
    return _EMBEDDER_BACKEND


def _embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    backend = _get_embedder_backend()
    if _EMBEDDER is not None:
        try:
            return [_norm_vector(list(vec)) for vec in _EMBEDDER.embed(texts)], backend
        except Exception as exc:  # optional model runtime can fail independently
            backend = f"hash-fallback:{type(exc).__name__}"
    return [_hash_embedding(text) for text in texts], backend


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _chunk_text(text: str, max_chars: int = VECTOR_CHUNK_CHARS, overlap: int = VECTOR_CHUNK_OVERLAP) -> list[dict[str, Any]]:
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    start_line = 1
    current_len = 0
    for line_no, line in enumerate(lines, start=1):
        add_len = len(line) + 1
        if current and current_len + add_len > max_chars:
            body = "\n".join(current).strip()
            if body:
                chunks.append({"text": body, "line": start_line})
            overlap_text = body[-overlap:] if overlap > 0 else ""
            current = [overlap_text, line] if overlap_text else [line]
            start_line = line_no
            current_len = sum(len(x) + 1 for x in current)
        else:
            if not current:
                start_line = line_no
            current.append(line)
            current_len += add_len
    body = "\n".join(current).strip()
    if body:
        chunks.append({"text": body, "line": start_line})
    return chunks or [{"text": text[:max_chars], "line": 1}]


def _vector_signature(paths: list[Path]) -> dict[str, Any]:
    files = []
    for path in paths:
        st = path.stat()
        files.append({"path": str(path.relative_to(REPO_PATH)), "mtime_ns": st.st_mtime_ns, "size": st.st_size})
    return {
        "repo": str(REPO_PATH),
        "model": EMBEDDING_MODEL_NAME,
        "chunk_chars": VECTOR_CHUNK_CHARS,
        "chunk_overlap": VECTOR_CHUNK_OVERLAP,
        "files": files,
    }


def _load_vector_index(directory: str = ".", force_rebuild: bool = False) -> dict[str, Any]:
    paths = list(_iter_markdown(directory) or [])
    signature = _vector_signature(paths)
    if not force_rebuild and VECTOR_INDEX_PATH.exists():
        try:
            cached = json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return cached
        except Exception:
            pass

    start = time.time()
    entries: list[dict[str, Any]] = []
    texts_to_embed: list[str] = []
    pending: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        fm = _frontmatter(text)
        rel = str(path.relative_to(REPO_PATH))
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        prefix = f"Title: {fm.get('title', path.stem)}\nPath: {rel}\nTags: {', '.join(map(str, tags))}\n\n"
        for idx, chunk in enumerate(_chunk_text(text)):
            item = {
                "path": rel,
                "chunk_id": idx,
                "line": chunk["line"],
                "title": fm.get("title", path.stem),
                "status": fm.get("status"),
                "tags": tags,
                "source_of_truth": fm.get("source_of_truth"),
                "text": chunk["text"][:VECTOR_CHUNK_CHARS + 200],
            }
            pending.append(item)
            texts_to_embed.append(prefix + chunk["text"])
    backend = _get_embedder_backend()
    batch_size = int(os.environ.get("BRAIN_EMBED_BATCH_SIZE", "8"))
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        vectors, batch_backend = _embed_texts(batch_texts)
        backend = batch_backend
        for item, vector in zip(pending[i:i + batch_size], vectors):
            entries.append({**item, "vector": vector})
    index = {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(time.time() - start, 3),
        "backend": backend,
        "entries": entries,
        "signature": signature,
    }
    VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTOR_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return index


def _best_vector_hits(query: str, directory: str, domain: str | None, tag: str | None, status: str | None, limit: int, force_rebuild: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = _load_vector_index(directory=directory, force_rebuild=force_rebuild)
    query_vec, backend = _embed_texts([query])
    if not query_vec:
        return [], {"backend": backend, "index_entries": len(index.get("entries", []))}
    qv = query_vec[0]
    by_doc: dict[str, dict[str, Any]] = {}
    for entry in index.get("entries", []):
        path = _safe_path(entry["path"])
        fm = {
            "status": entry.get("status"),
            "tags": entry.get("tags", []),
            "source_of_truth": entry.get("source_of_truth"),
        }
        if not _matches_filters(path, fm, domain, status, tag):
            continue
        sim = _cosine(qv, entry.get("vector", []))
        doc = by_doc.get(entry["path"])
        if doc is None or sim > doc["vector_score"]:
            by_doc[entry["path"]] = {
                "path": entry["path"],
                "vector_score": sim,
                "title": entry.get("title"),
                "status": entry.get("status"),
                "tags": entry.get("tags", []),
                "source_of_truth": entry.get("source_of_truth"),
                "vector_snippet": {"line": entry.get("line", 1), "snippet": entry.get("text", "")[:1200]},
            }
    hits = sorted(by_doc.values(), key=lambda r: r["vector_score"], reverse=True)[:limit]
    return hits, {
        "backend": backend,
        "index_entries": len(index.get("entries", [])),
        "built_at": index.get("built_at"),
        "build_seconds": index.get("build_seconds"),
        "index_path": str(VECTOR_INDEX_PATH),
    }


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


def _pull_ff() -> None:
    _run_git(["pull", "--ff-only", "origin", "main"], check=True)


def _validate_after_write(changed_docs: bool) -> list[str]:
    # ponytail: validator/indexer scripts are optional. Original company-brain ships them;
    # arbitrary user brains don't. Skip silently when absent.
    messages = []
    gen = REPO_PATH / "scripts" / "generate_index.py"
    val = REPO_PATH / "scripts" / "validate_docs.py"
    if changed_docs and gen.exists():
        _run_repo_cmd(["python3", "scripts/generate_index.py"], check=True)
        messages.append("regenerated docs/index.md")
    if val.exists():
        _run_repo_cmd(["python3", "scripts/validate_docs.py"], check=True)
        messages.append("validate_docs passed")
    if gen.exists():
        _run_repo_cmd(["python3", "scripts/generate_index.py", "--check"], check=True)
        messages.append("generated index check passed")
    return messages


def _commit_push(paths: list[str], commit_message: str) -> dict[str, Any]:
    _run_git(["add", *paths], check=True)
    diff_cached = _run_git(["diff", "--cached", "--quiet"])
    if diff_cached.returncode == 0:
        return {"changed": False, "message": "No changes to commit."}
    _run_git(["commit", "-m", commit_message], check=True)
    _run_git(["push", "origin", "main"], check=True)
    status = _run_git(["status", "--short"], check=True).stdout.strip()
    head = _run_git(["rev-parse", "--short", "HEAD"], check=True).stdout.strip()
    return {"changed": True, "pushed": True, "clean": status == "", "dirty_status": status, "head": head}


@mcp.tool(name="brain_list")
def brain_list(directory: str = ".", max_results: int = 500) -> str:
    """List markdown files in the brain."""
    _ensure_repo()
    files = []
    for path in _iter_markdown(directory) or []:
        files.append(str(path.relative_to(REPO_PATH)))
        if len(files) >= max_results:
            break
    return _json({"repo": str(REPO_PATH), "files": files, "truncated": len(files) >= max_results})


@mcp.tool(name="brain_search")
def brain_search(
    query: str,
    directory: str = ".",
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
        # Pull more vector docs than requested before fusion so vector-only
        # candidates can compete with high lexical matches.
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
        return _json({"query": query, "tokens": toks, "mode": mode, "rerank": rerank_meta, "results": results[:max_results], "truncated": len(results) > max_results})

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
        return _json({"query": query, "tokens": toks, "mode": mode, "vector": vector_meta, "rerank": rerank_meta, "results": results[:max_results], "truncated": len(results) > max_results})

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
    return _json({
        "query": query,
        "tokens": toks,
        "mode": mode,
        "weights": {"keyword": HYBRID_KEYWORD_WEIGHT, "vector": HYBRID_VECTOR_WEIGHT},
        "vector": vector_meta,
        "rerank": rerank_meta,
        "results": results[:max_results],
        "truncated": len(results) > max_results,
    })


@mcp.tool(name="brain_read")
def brain_read(file_path: str) -> str:
    """Read a specific brain file by repo-relative path."""
    _ensure_repo()
    full = _safe_path(file_path)
    if not full.exists() or not full.is_file():
        return _json({"error": f"not found: {file_path}"})
    content = full.read_text(encoding="utf-8")
    return _json({
        "path": str(full.relative_to(REPO_PATH)),
        "frontmatter": _frontmatter(content),
        "content": content[:MAX_READ_CHARS],
        "truncated": len(content) > MAX_READ_CHARS,
    })


@mcp.tool(name="brain_answer")
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
    brain_retrieval_log(question=question, hits=[u["path"] for u in used], used_docs=[u["path"] for u in used], outcome="evidence_bundle")
    return _json({"question": question, "evidence": used, "answer_instruction": "Use only this evidence unless you call more tools. Cite paths in the final answer."})


@mcp.tool(name="brain_write")
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
    old = full.read_text(encoding="utf-8") if full.exists() else ""
    full.parent.mkdir(parents=True, exist_ok=True)
    new = content.rstrip() + "\n" if mode == "replace" else old.rstrip() + "\n" + content.rstrip() + "\n"
    if _detect_secrets(new):
        return _json({"error": "Refusing to write because resulting file appears to contain secrets."})
    full.write_text(new, encoding="utf-8")
    changed_docs = rel.startswith("docs/")
    validation = _validate_after_write(changed_docs=changed_docs)
    paths = [rel]
    if changed_docs:
        paths.append("docs/index.md")
    result = _commit_push(paths, commit_message)
    return _json({"path": rel, "validation": validation, **result})


@mcp.tool(name="brain_audit")
def brain_audit() -> str:
    """Summarize repo health for retrieval and write safety."""
    _ensure_repo()
    status = _run_git(["status", "--short"], check=True).stdout.strip()
    val_script = REPO_PATH / "scripts" / "validate_docs.py"
    gen_script = REPO_PATH / "scripts" / "generate_index.py"
    validate = _run_repo_cmd(["python3", "scripts/validate_docs.py"]).stdout.strip() if val_script.exists() else "(no validator)"
    index = _run_repo_cmd(["python3", "scripts/generate_index.py", "--check"]).stdout.strip() if gen_script.exists() else "(no indexer)"
    docs = list(_iter_markdown(".") or [])
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
def brain_classify(text: str) -> str:
    """Heuristically route text to brain, personal brain, memory, nowhere, or split."""
    lower = text.lower()
    if _detect_secrets(text):
        return _json({"destination": "nowhere", "reason": "Looks like secret/credential material."})
    company_terms = ["pmxt", "customer", "client", "investor", "yc", "martin", "sor", "prediction market", "supabase", "clickhouse", "company", "venue", "api"]
    personal_terms = ["family", "health", "home", "travel", "diet", "personal", "private"]
    if any(k in lower for k in company_terms) and any(k in lower for k in personal_terms):
        dest = "split"
        reason = "Contains both company and personal indicators; write only team-shareable subset to brain."
    elif any(k in lower for k in company_terms):
        dest = "company_brain"
        reason = "Contains PMXT/company/work indicators."
    elif any(k in lower for k in ["prefer", "timezone", "style", "always"]):
        dest = "assistant_memory_or_personal_brain"
        reason = "Looks like a durable user preference/profile fact."
    else:
        dest = "personal_or_ignore"
        reason = "No clear company indicator; avoid shared brain unless work relevance is explicit."
    return _json({"destination": dest, "reason": reason})


@mcp.tool(name="brain_retrieval_log")
def brain_retrieval_log(question: str, hits: list[str] | None = None, used_docs: list[str] | None = None, outcome: str = "unknown") -> str:
    """Append retrieval telemetry to a local JSONL log for later outcome review."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "hits": hits or [],
        "used_docs": used_docs or [],
        "outcome": outcome,
    }
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _json({"logged": True, "path": str(LOG_PATH), "record": record})


if __name__ == "__main__":
    if os.environ.get("BRAIN_MCP_SELF_TEST") == "1":
        print(brain_audit())
    else:
        mcp.run(transport="stdio")
