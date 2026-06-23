"""Interactive setup wizard. Hermes-style sections registry + questionary prompts."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import questionary

from . import mcp_snippet
from ._user_config import CONFIG_DIR, CONFIG_PATH, BreinConfig, load, save


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    run: Callable[[BreinConfig], BreinConfig]


def _expand(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def setup_repo(cfg: BreinConfig) -> BreinConfig:
    mode = questionary.select(
        "Brain repo:",
        choices=["Use existing path", "Clone from git", "Create new empty repo"],
        default="Use existing path" if cfg.repo_path else "Create new empty repo",
    ).ask()
    if mode is None:
        return cfg

    if mode == "Use existing path":
        path = questionary.path(
            "Path to existing brain repo:",
            default=cfg.repo_path or str(Path.home() / ".braincorp" / "brain"),
            only_directories=True,
        ).ask()
        if not path:
            return cfg
        cfg.repo_path = _expand(path)

    elif mode == "Clone from git":
        url = questionary.text("Git URL:").ask()
        dest = questionary.path(
            "Clone into:",
            default=str(Path.home() / ".braincorp" / "brain"),
        ).ask()
        if not (url and dest):
            return cfg
        dest = _expand(dest)
        subprocess.run(["git", "clone", url, dest], check=True)
        cfg.repo_path = dest

    elif mode == "Create new empty repo":
        dest = questionary.path(
            "Create repo at:",
            default=cfg.repo_path or str(Path.home() / ".braincorp" / "brain"),
        ).ask()
        if not dest:
            return cfg
        dest_path = Path(_expand(dest))
        (dest_path / "docs").mkdir(parents=True, exist_ok=True)
        if not (dest_path / ".git").exists():
            subprocess.run(["git", "init", "-q", str(dest_path)], check=True)
        cfg.repo_path = str(dest_path)

    return cfg


def setup_paths(cfg: BreinConfig) -> BreinConfig:
    log = questionary.path(
        "Retrieval log path:",
        default=cfg.retrieval_log or str(CONFIG_DIR / "retrieval-log.jsonl"),
    ).ask()
    idx = questionary.path(
        "Vector index path:",
        default=cfg.vector_index or str(CONFIG_DIR / "vector-index.json"),
    ).ask()
    if log:
        cfg.retrieval_log = _expand(log)
        Path(cfg.retrieval_log).parent.mkdir(parents=True, exist_ok=True)
    if idx:
        cfg.vector_index = _expand(idx)
        Path(cfg.vector_index).parent.mkdir(parents=True, exist_ok=True)
    return cfg


def setup_vector(cfg: BreinConfig) -> BreinConfig:
    try:
        import fastembed  # noqa: F401
        fastembed_ok = True
    except ImportError:
        fastembed_ok = False

    if not fastembed_ok:
        questionary.print(
            "  fastembed not installed — embeddings will use the hash fallback.",
            style="fg:#888888",
        )
    model = questionary.text(
        "Embedding model:",
        default=cfg.embedding_model,
    ).ask()
    if model:
        cfg.embedding_model = model
    return cfg


def setup_eval(cfg: BreinConfig) -> BreinConfig:
    enabled = questionary.confirm(
        "Enable retrieval eval?",
        default=cfg.eval_enabled,
    ).ask()
    cfg.eval_enabled = bool(enabled)
    if cfg.eval_enabled:
        order = questionary.checkbox(
            "Host CLI fallback order (space to select, enter to confirm):",
            choices=[
                questionary.Choice(c, checked=(c in cfg.eval_host_order))
                for c in ("claude", "codex", "gemini")
            ],
        ).ask()
        if order:
            cfg.eval_host_order = order
    return cfg


def setup_mcp(cfg: BreinConfig) -> BreinConfig:
    if not cfg.repo_path:
        questionary.print(
            "  Skipping — run `brein setup repo` first.", style="fg:#cc8800"
        )
        return cfg
    client = questionary.select(
        "Generate MCP snippet for which client?",
        choices=[*mcp_snippet.CLIENTS, "skip"],
    ).ask()
    if not client or client == "skip":
        return cfg
    print()
    print(mcp_snippet.snippet(cfg, client))
    print()
    questionary.print(
        f"  Paste the block above into your {client} MCP config.",
        style="fg:#888888",
    )
    return cfg


SECTIONS: tuple[Section, ...] = (
    Section("repo",   "Brain repo location",         setup_repo),
    Section("paths",  "Log & vector index paths",    setup_paths),
    Section("vector", "Embeddings",                  setup_vector),
    Section("eval",   "Retrieval eval",              setup_eval),
    Section("mcp",    "MCP client snippet",          setup_mcp),
)


def _print_noninteractive_guidance() -> None:
    print(
        "brein setup is interactive and stdin is not a TTY.\n"
        "For headless setup, write the config file directly:\n"
        f"  {CONFIG_PATH}\n"
        "Required fields: repo_path (absolute path to your git-backed brain repo).\n"
        "See README for the full schema.",
        file=sys.stderr,
    )


def run(section: str | None = None) -> int:
    if not sys.stdin.isatty():
        _print_noninteractive_guidance()
        return 2

    valid = {s.key for s in SECTIONS}
    if section is not None and section not in valid:
        print(f"unknown section {section!r}. valid: {sorted(valid)}", file=sys.stderr)
        return 2

    cfg = load()
    sections = [s for s in SECTIONS if section in (None, s.key)]
    for s in sections:
        questionary.print(f"\n── {s.title} ──", style="bold")
        try:
            cfg = s.run(cfg)
        except KeyboardInterrupt:
            print("\naborted.", file=sys.stderr)
            return 130

    save(cfg)
    questionary.print(f"\n✓ saved {CONFIG_PATH}", style="fg:#00aa66 bold")
    return 0
