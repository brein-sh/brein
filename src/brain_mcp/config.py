"""Environment-driven configuration for the brain MCP server."""

from __future__ import annotations

import os
import re
from pathlib import Path

_HOME = Path.home()
REPO_PATH = Path(os.environ.get("BRAIN_REPO", str(_HOME / ".brein" / "brain"))).resolve()
DOCS_PATH = REPO_PATH / "docs"
MAX_READ_CHARS = int(os.environ.get("BRAIN_MAX_READ_CHARS", "80000"))
LOG_PATH = Path(os.environ.get("BRAIN_RETRIEVAL_LOG", str(REPO_PATH / "telemetry" / "retrieval-log.jsonl")))
VECTOR_INDEX_PATH = Path(os.environ.get("BRAIN_VECTOR_INDEX", str(_HOME / ".brein" / "vector-index.json")))
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

SECRET_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd|api[_-]?key|x-api-key|secret|token|auth[_-]?token|private[_-]?key)\s*[:=]\s*[^\s`'\"]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"(?i)seed phrase\s*[:=]"),
    # AWS access key id (AKIA = long-term IAM, ASIA = temp STS).
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
]

ALLOWED_WRITE_PREFIXES = ("docs/", "skills/", "templates/")
ALLOWED_ROOT_WRITES = {"AGENTS.md", "README.md", "CONTRIBUTING.md"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "our", "the", "to", "what", "when",
    "where", "who", "why", "with", "does", "do", "we", "have", "has",
    "against", "about",
}
