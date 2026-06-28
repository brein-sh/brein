"""Embedding + incremental vector index for the brain."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    EMBEDDING_MODEL_NAME,
    HASH_EMBED_DIMS,
    REPO_PATH,
    VECTOR_CHUNK_CHARS,
    VECTOR_CHUNK_OVERLAP,
    VECTOR_INDEX_PATH,
)
from .shared import _frontmatter, _iter_markdown, _matches_filters, _safe_path

_EMBEDDER = None
_EMBEDDER_BACKEND = "uninitialized"


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


def _vector_health() -> dict[str, Any]:
    """Return current vector backend health, including whether semantic search is real."""
    backend = _get_embedder_backend()
    is_fallback = backend.startswith("hash-fallback")
    return {
        "backend": backend,
        "degraded": is_fallback,
        "warning": (
            "Vector backend is hash-fallback — semantic recall is significantly degraded. "
            "Install/repair `fastembed` to restore real embeddings."
        ) if is_fallback else None,
    }


def _embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    backend = _get_embedder_backend()
    if _EMBEDDER is not None:
        try:
            # ponytail: no `parallel=` → in-process ONNX runtime threading.
            # Any parallel=N triggers multiprocessing which fork-bombs without a __main__ guard.
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


def _global_signature() -> dict[str, Any]:
    return {
        "repo": str(REPO_PATH),
        "model": EMBEDDING_MODEL_NAME,
        "chunk_chars": VECTOR_CHUNK_CHARS,
        "chunk_overlap": VECTOR_CHUNK_OVERLAP,
    }


def _file_fingerprint(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _vector_signature(paths: list[Path]) -> dict[str, Any]:
    # ponytail: kept for any external caller; load path no longer uses it
    sig = _global_signature()
    sig["files"] = [{"path": str(p.relative_to(REPO_PATH)), **_file_fingerprint(p)} for p in paths]
    return sig


# ponytail: in-memory cache of the parsed index. JSON parse of the 1k-chunk
# blob was the dominant per-call cost. Invalidate on global-sig or per-file
# fingerprint change — both cheap to compute (stat per file).
_INDEX_CACHE: dict[str, Any] = {"index": None, "global_sig": None, "fps": None, "directory": None}


def _load_vector_index(directory: str = "docs", force_rebuild: bool = False, progress_cb=None) -> dict[str, Any]:
    paths = list(_iter_markdown(directory) or [])
    current_fps = {str(p.relative_to(REPO_PATH)): _file_fingerprint(p) for p in paths}
    global_sig = _global_signature()

    if (
        not force_rebuild
        and _INDEX_CACHE["index"] is not None
        and _INDEX_CACHE["directory"] == directory
        and _INDEX_CACHE["global_sig"] == global_sig
        and _INDEX_CACHE["fps"] == current_fps
    ):
        return _INDEX_CACHE["index"]

    cached_entries_by_path: dict[str, list[dict[str, Any]]] = {}
    cached_fps: dict[str, dict[str, int]] = {}
    if not force_rebuild and VECTOR_INDEX_PATH.exists():
        try:
            cached = json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))
            # Treat non-dict top-level JSON (e.g. a list) as corrupted: fall
            # through to a clean rebuild instead of crashing on .get().
            if isinstance(cached, dict) and cached.get("global_signature") == global_sig:
                cached_fps = cached.get("file_signatures", {}) or {}
                for entry in cached.get("entries", []):
                    cached_entries_by_path.setdefault(entry["path"], []).append(entry)
        except Exception:
            pass

    start = time.time()
    entries: list[dict[str, Any]] = []
    texts_to_embed: list[str] = []
    pending: list[dict[str, Any]] = []
    reused = 0
    reembedded = 0
    for path in paths:
        rel = str(path.relative_to(REPO_PATH))
        if cached_fps.get(rel) == current_fps[rel] and rel in cached_entries_by_path:
            entries.extend(cached_entries_by_path[rel])
            reused += len(cached_entries_by_path[rel])
            continue
        reembedded += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        fm = _frontmatter(text)
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
    total = len(texts_to_embed)
    if _EMBEDDER is not None and total > 0:
        # ponytail: in-process ONNX threading (no `parallel=`). Stream the
        # generator so we don't materialize every vector at once.
        done = 0
        for item, vec in zip(pending, _EMBEDDER.embed(texts_to_embed)):
            entries.append({**item, "vector": _norm_vector(list(vec))})
            done += 1
            if progress_cb and (done % 64 == 0 or done == total):
                progress_cb(done, total)
    else:
        # Hash fallback path (no fastembed) — small loop, no parallel benefit
        batch_size = int(os.environ.get("BRAIN_EMBED_BATCH_SIZE", "128"))
        for i in range(0, total, batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            vectors, batch_backend = _embed_texts(batch_texts)
            backend = batch_backend
            for item, vector in zip(pending[i:i + batch_size], vectors):
                entries.append({**item, "vector": vector})
            if progress_cb:
                progress_cb(min(i + batch_size, total), total)
    index = {
        "version": 2,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(time.time() - start, 3),
        "backend": backend,
        "entries": entries,
        "global_signature": global_sig,
        "file_signatures": current_fps,
        "incremental_stats": {
            "reused_entries": reused,
            "reembedded_files": reembedded,
            "total_files": len(paths),
        },
        "signature": {**global_sig, "files": [{"path": p, **fp} for p, fp in current_fps.items()]},
    }
    try:
        VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        VECTOR_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Vector index could not be written to {VECTOR_INDEX_PATH}: {exc}. "
            f"Set BRAIN_VECTOR_INDEX to a writable path."
        ) from exc
    _INDEX_CACHE["index"] = index
    _INDEX_CACHE["global_sig"] = global_sig
    _INDEX_CACHE["fps"] = current_fps
    _INDEX_CACHE["directory"] = directory
    return index


_RECENCY_CACHE: dict[str, str | None] = {}


def _doc_date_str(rel: str) -> str | None:
    """Pull `last_reviewed` or `decided` from a doc's frontmatter.

    Cached per process — frontmatter parse is cheap but we hit this on every
    top-K rerank pass. Stored even when None to avoid re-reading missing fields.
    """
    if rel in _RECENCY_CACHE:
        return _RECENCY_CACHE[rel]
    try:
        full = _safe_path(rel)
        text = full.read_text(encoding="utf-8")
        fm = _frontmatter(text)
        date = fm.get("last_reviewed") or fm.get("decided")
        date_str = str(date).strip() if date else None
    except Exception:
        date_str = None
    _RECENCY_CACHE[rel] = date_str
    return date_str


def _post_rank_boost(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic tiebreaker pass over vector hits.

    Why: the 13% no-brain-win bucket in our A/B eval is dominated by narrative
    notes ranking equal-or-above the canonical decision doc that arbitrates a
    topic. Vector similarity alone can't tell which doc *settles* the question.
    We nudge — not replace — vector order using:

        source_of_truth: true      → +0.05
        recent last_reviewed/decided → up to +0.02 (linear over ~3 years)

    Bumps are small enough that a clearly-better vector hit still wins. Adds
    `boosted_score` so the caller can see what happened; `vector_score` stays
    untouched for telemetry parity.
    """
    if not hits:
        return hits
    today = datetime.now(timezone.utc).date()
    boosted: list[dict[str, Any]] = []
    for hit in hits:
        bonus = 0.0
        signals: dict[str, Any] = {}
        if str(hit.get("source_of_truth", "")).lower() == "true":
            bonus += 0.05
            signals["source_of_truth"] = True
        date_str = _doc_date_str(hit["path"])
        if date_str:
            try:
                doc_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                age_days = (today - doc_date).days
                # Linear decay: same-day → +0.02, 3y old → 0. Older → 0 (no penalty).
                recency = max(0.0, 0.02 * (1.0 - age_days / 1095.0))
                if recency > 0:
                    bonus += recency
                    signals["recency"] = round(recency, 4)
            except ValueError:
                pass
        new = dict(hit)
        new["boost"] = round(bonus, 6)
        new["boost_signals"] = signals
        new["vector_score"] = hit["vector_score"] + bonus
        boosted.append(new)
    boosted.sort(key=lambda r: r["vector_score"], reverse=True)
    return boosted


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
    hits = _post_rank_boost(hits)
    health = _vector_health()
    return hits, {
        "backend": backend,
        "degraded": health["degraded"],
        "warning": health["warning"],
        "index_entries": len(index.get("entries", [])),
        "built_at": index.get("built_at"),
        "build_seconds": index.get("build_seconds"),
        "index_path": str(VECTOR_INDEX_PATH),
    }
