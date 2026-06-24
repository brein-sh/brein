"""Interactive setup wizard. Hermes-style sections registry + questionary prompts."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import questionary

from . import mcp_install, mcp_snippet
from ._user_config import CONFIG_DIR, CONFIG_PATH, BreinConfig, load, save


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    run: Callable[[BreinConfig], BreinConfig]


def _expand(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _is_non_empty_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())


def _resolve_existing_dest(dest: Path, *, label: str) -> str | None:
    """If dest exists and is non-empty, ask the user what to do.
    Returns the path to use, or None to abort this branch."""
    if not _is_non_empty_dir(dest):
        return str(dest)
    is_repo = (dest / ".git").exists()
    msg = f"{dest} already exists and is non-empty"
    msg += " (looks like a git repo)." if is_repo else "."
    questionary.print(f"  {msg}", style="fg:#cc8800")
    choices = ["Use it as-is", "Pick a different path", "Abort"]
    pick = questionary.select(f"What to do for {label}?", choices=choices).ask()
    if pick == "Use it as-is":
        return str(dest)
    if pick == "Pick a different path":
        new = questionary.path(f"New path for {label}:").ask()
        return _expand(new) if new else None
    return None


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
            default=cfg.repo_path or str(Path.home() / ".brein" / "brain"),
            only_directories=True,
        ).ask()
        if not path:
            return cfg
        cfg.repo_path = _expand(path)

    elif mode == "Clone from git":
        url = questionary.text("Git URL:").ask()
        dest_input = questionary.path(
            "Clone into:",
            default=str(Path.home() / ".brein" / "brain"),
        ).ask()
        if not (url and dest_input):
            return cfg
        resolved = _resolve_existing_dest(Path(_expand(dest_input)), label="clone target")
        if resolved is None:
            return cfg
        if _is_non_empty_dir(Path(resolved)):
            # User chose "Use it as-is" — skip the clone.
            cfg.repo_path = resolved
        else:
            try:
                subprocess.run(["git", "clone", url, resolved], check=True)
            except subprocess.CalledProcessError as e:
                questionary.print(f"  git clone failed (exit {e.returncode})", style="fg:#cc0000")
                return cfg
            cfg.repo_path = resolved

    elif mode == "Create new empty repo":
        dest_input = questionary.path(
            "Create repo at:",
            default=cfg.repo_path or str(Path.home() / ".brein" / "brain"),
        ).ask()
        if not dest_input:
            return cfg
        dest_path = Path(_expand(dest_input))
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / "docs").mkdir(parents=True, exist_ok=True)
        if not (dest_path / ".git").exists():
            try:
                subprocess.run(["git", "init", "-q", str(dest_path)], check=True)
            except subprocess.CalledProcessError as e:
                questionary.print(f"  git init failed (exit {e.returncode})", style="fg:#cc0000")
                return cfg
        cfg.repo_path = str(dest_path)

    return cfg


def setup_paths(cfg: BreinConfig) -> BreinConfig:
    # Retrieval log defaults inside the brain repo — that's the only place
    # telemetry auto-commit works. Vector index stays outside; it's a
    # recomputable cache that would bloat the repo.
    log_default = cfg.retrieval_log or (
        str(Path(cfg.repo_path) / "telemetry" / "retrieval-log.jsonl")
        if cfg.repo_path
        else str(CONFIG_DIR / "retrieval-log.jsonl")
    )
    idx_default = cfg.vector_index or str(CONFIG_DIR / "vector-index.json")

    log = questionary.path("Retrieval log path:", default=log_default).ask()
    idx = questionary.path("Vector index path (cache):", default=idx_default).ask()
    if log:
        cfg.retrieval_log = _expand(log)
        Path(cfg.retrieval_log).parent.mkdir(parents=True, exist_ok=True)
    if idx:
        cfg.vector_index = _expand(idx)
        Path(cfg.vector_index).parent.mkdir(parents=True, exist_ok=True)
    return cfg


EMBEDDING_MODELS = [
    ("BAAI/bge-small-en-v1.5", "English, fast, 384 dims (default)"),
    ("BAAI/bge-base-en-v1.5", "English, better recall, 768 dims"),
    ("BAAI/bge-large-en-v1.5", "English, best recall, 1024 dims, slow"),
    ("intfloat/multilingual-e5-small", "Multilingual, 384 dims"),
]
_OTHER = "Other (type model name)"


def setup_vector(cfg: BreinConfig) -> BreinConfig:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        questionary.print(
            "  fastembed not installed — embeddings will use the hash fallback.",
            style="fg:#888888",
        )

    choices = [questionary.Choice(f"{m}  — {desc}", value=m) for m, desc in EMBEDDING_MODELS]
    choices.append(questionary.Choice(_OTHER, value=_OTHER))
    default = next((c for c in choices if c.value == cfg.embedding_model), choices[0])

    pick = questionary.select("Embedding model:", choices=choices, default=default).ask()
    if pick is None:
        return cfg
    if pick == _OTHER:
        custom = questionary.text("Model name:", default=cfg.embedding_model).ask()
        if custom:
            cfg.embedding_model = custom
    else:
        cfg.embedding_model = pick
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

    server = json.loads(mcp_snippet.snippet(cfg, "generic"))["mcpServers"]["brain"]
    detected = mcp_install.detect_installed()

    if detected:
        names = ", ".join(c.label for c in detected)
        questionary.print(f"  Detected: {names}", style="fg:#888888")
        if questionary.confirm("Install brein to all detected clients?", default=True).ask():
            restart_notes: list[str] = []
            for c in detected:
                r = c.install(server)
                if r.ok:
                    questionary.print(f"  ✓ {c.label} — {r.detail}", style="fg:#00aa66")
                    if c.restart_note:
                        restart_notes.append(c.restart_note)
                else:
                    questionary.print(f"  ✗ {c.label} — {r.detail}", style="fg:#cc0000")
            if restart_notes:
                questionary.print(f"\n  Next: {'; '.join(restart_notes)}", style="fg:#888888")
            return cfg

    # No detected clients, or user declined auto-install. Print for manual paste.
    print()
    print(mcp_snippet.snippet(cfg, "generic"))
    print()
    questionary.print(
        "  Paste the block above into your client's MCP config.",
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
