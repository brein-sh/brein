"""Health checks: ok/warn/fail with actionable remediation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ._user_config import CONFIG_PATH, BreinConfig, load

OK, WARN, FAIL = "ok", "warn", "fail"
_GLYPH = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}


@dataclass
class Result:
    status: str
    name: str
    detail: str = ""
    fix: str = ""


Check = Callable[[BreinConfig], Result]


def check_config_file(_: BreinConfig) -> Result:
    if CONFIG_PATH.exists():
        return Result(OK, "config file", str(CONFIG_PATH))
    return Result(FAIL, "config file", f"missing {CONFIG_PATH}", fix="run `brein setup`")


def check_python(_: BreinConfig) -> Result:
    v = sys.version_info
    if v >= (3, 11):
        return Result(OK, "python", f"{v.major}.{v.minor}.{v.micro}")
    return Result(FAIL, "python", f"{v.major}.{v.minor} (need >=3.11)")


def check_uv(_: BreinConfig) -> Result:
    if shutil.which("uv"):
        return Result(OK, "uv", "available")
    return Result(WARN, "uv", "not on PATH", fix="install from https://docs.astral.sh/uv/")


def check_repo(cfg: BreinConfig) -> Result:
    if not cfg.repo_path:
        return Result(FAIL, "brain repo", "not configured", fix="run `brein setup repo`")
    p = Path(cfg.repo_path)
    if not p.exists():
        return Result(FAIL, "brain repo", f"{p} does not exist", fix="run `brein setup repo`")
    if not (p / ".git").exists():
        return Result(WARN, "brain repo", f"{p} is not a git repo", fix=f"cd {p} && git init")
    if not (p / "docs").exists():
        return Result(WARN, "brain repo", f"{p}/docs missing", fix=f"mkdir -p {p}/docs")
    return Result(OK, "brain repo", str(p))


def check_git_clean(cfg: BreinConfig) -> Result:
    if not cfg.repo_path or not (Path(cfg.repo_path) / ".git").exists():
        return Result(WARN, "git status", "skipped (no repo)")
    out = subprocess.run(
        ["git", "-C", cfg.repo_path, "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return Result(FAIL, "git status", out.stderr.strip())
    if out.stdout.strip():
        return Result(WARN, "git status", "working tree has uncommitted changes")
    return Result(OK, "git status", "clean")


def check_vector_backend(_: BreinConfig) -> Result:
    try:
        import fastembed  # noqa: F401
        return Result(OK, "vector backend", "fastembed")
    except ImportError:
        return Result(
            WARN, "vector backend", "fastembed missing — using hash fallback",
            fix="uv sync (or `pip install fastembed`)",
        )


def check_writable_paths(cfg: BreinConfig) -> Result:
    problems = []
    for label, raw in (("retrieval_log", cfg.retrieval_log), ("vector_index", cfg.vector_index)):
        if not raw:
            continue
        parent = Path(raw).parent
        if not parent.exists():
            problems.append(f"{label}: {parent} missing")
        elif not _writable(parent):
            problems.append(f"{label}: {parent} not writable")
    if problems:
        return Result(FAIL, "writable paths", "; ".join(problems), fix="run `brein setup paths`")
    return Result(OK, "writable paths", "ok")


def _writable(p: Path) -> bool:
    import os
    return os.access(p, os.W_OK)


def check_mcp_server(_: BreinConfig) -> Result:
    bin_path = shutil.which("brain-mcp")
    if bin_path:
        return Result(OK, "brain-mcp launcher", bin_path)
    return Result(
        WARN, "brain-mcp launcher", "console script not on PATH",
        fix="run `uv sync` or `pip install -e .` from the repo",
    )


CHECKS: tuple[Check, ...] = (
    check_python,
    check_uv,
    check_config_file,
    check_repo,
    check_git_clean,
    check_vector_backend,
    check_writable_paths,
    check_mcp_server,
)


def run() -> int:
    cfg = load()
    worst = OK
    rank = {OK: 0, WARN: 1, FAIL: 2}
    for check in CHECKS:
        try:
            r = check(cfg)
        except Exception as e:  # ponytail: check bugs shouldn't crash the report
            r = Result(FAIL, check.__name__, f"check raised: {e!r}")
        line = f"  {_GLYPH[r.status]} {r.name:<22} {r.detail}"
        print(line)
        if r.fix and r.status != OK:
            print(f"    → {r.fix}")
        if rank[r.status] > rank[worst]:
            worst = r.status
    return 0 if worst != FAIL else 1
