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

from config import (
    EMBEDDING_MODEL_NAME,
    HASH_EMBED_DIMS,
    REPO_PATH,
    VECTOR_CHUNK_CHARS,
    VECTOR_CHUNK_OVERLAP,
    VECTOR_INDEX_PATH,
)
from shared import _frontmatter, _iter_markdown, _matches_filters, _safe_path

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


def _load_vector_index(directory: str = "docs", force_rebuild: bool = False) -> dict[str, Any]:
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
            if cached.get("global_signature") == global_sig:
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
    batch_size = int(os.environ.get("BRAIN_EMBED_BATCH_SIZE", "8"))
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        vectors, batch_backend = _embed_texts(batch_texts)
        backend = batch_backend
        for item, vector in zip(pending[i:i + batch_size], vectors):
            entries.append({**item, "vector": vector})
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
