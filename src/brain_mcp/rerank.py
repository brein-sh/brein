"""Rerank candidates from hybrid retrieval via LLM (default) or local heuristic."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import (
    RERANK_BIN_DEFAULT,
    RERANK_COMMAND_DEFAULT,
    RERANK_MAX_TOP_K,
    RERANK_MODEL_DEFAULT,
    RERANK_PROVIDER_DEFAULT,
    RERANK_SNIPPET_CHARS,
    RERANK_SNIPPET_COUNT,
    RERANK_TIMEOUT_SECONDS,
)
from .shared import _result_text, _truncate_text, _truthy


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
        "You are a strict document reranker for an org-wide knowledge brain.\n"
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
