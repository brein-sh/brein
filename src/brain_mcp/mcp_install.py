"""Per-MCP-client install helpers.

Each Client has a `detect()` (is it installed?) and an `install(server_block)`
that writes the brein server block into the client's config in-place.
JSON-file clients back up the existing file before writing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import tomli_w


@dataclass
class InstallResult:
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Client:
    key: str
    label: str
    detect: Callable[[], bool]
    install: Callable[[dict], InstallResult]
    restart_note: str = ""


# ── Claude Code (CLI) ────────────────────────────────────────────────────────

def _claude_code_detect() -> bool:
    return shutil.which("claude") is not None


def _claude_code_install(server: dict) -> InstallResult:
    # Remove first to make the operation idempotent — `add-json` errors if the
    # name already exists. Ignore remove failure (means it wasn't there).
    subprocess.run(
        ["claude", "mcp", "remove", "brain", "--scope", "user"],
        capture_output=True, text=True,
    )
    r = subprocess.run(
        ["claude", "mcp", "add-json", "brain", json.dumps(server), "--scope", "user"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return InstallResult(False, r.stderr.strip() or r.stdout.strip())
    cleared = _clear_brain_from_disabled_lists()
    detail = "added via `claude mcp add-json --scope user`"
    if cleared:
        detail += f" (cleared {cleared} per-project disable)"
    return InstallResult(True, detail)


def _clear_brain_from_disabled_lists() -> int:
    """Remove 'brain' from every projects.<X>.disabledMcpServers list in
    ~/.claude.json. Otherwise a prior user-disable in /mcp keeps brain off
    after install. Returns the number of projects touched."""
    path = Path.home() / ".claude.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return 0  # silently skip; we never destructively edit broken state
    projects = data.get("projects")
    if not isinstance(projects, dict):
        return 0
    changed = 0
    for proj in projects.values():
        if not isinstance(proj, dict):
            continue
        disabled = proj.get("disabledMcpServers")
        if isinstance(disabled, list) and "brain" in disabled:
            proj["disabledMcpServers"] = [s for s in disabled if s != "brain"]
            changed += 1
    if not changed:
        return 0
    shutil.copy2(path, path.with_suffix(".json.bak"))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
    return changed


# ── Claude Desktop (JSON file) ───────────────────────────────────────────────

def _claude_desktop_path() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", str(home))) / "Claude" / "claude_desktop_config.json"
    return home / ".config" / "Claude" / "claude_desktop_config.json"


def _claude_desktop_detect() -> bool:
    return _claude_desktop_path().parent.exists()


# ── Cursor (JSON file) ───────────────────────────────────────────────────────

def _cursor_path() -> Path:
    return Path.home() / ".cursor" / "mcp.json"


def _cursor_detect() -> bool:
    return _cursor_path().parent.exists()


# ── Generic JSON merger ──────────────────────────────────────────────────────

def _merge_mcp_json(path: Path, server: dict) -> InstallResult:
    """Write `server` under mcpServers.brain in a JSON config file. Backs up first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return InstallResult(False, f"existing file is invalid JSON: {path}")
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    else:
        data = {}
    if not isinstance(data, dict):
        return InstallResult(False, f"existing root is not an object: {path}")
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        return InstallResult(False, f"existing mcpServers is not an object: {path}")
    servers["brain"] = server
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
    return InstallResult(True, f"wrote {path}")


def _claude_desktop_install(server: dict) -> InstallResult:
    return _merge_mcp_json(_claude_desktop_path(), server)


def _cursor_install(server: dict) -> InstallResult:
    return _merge_mcp_json(_cursor_path(), server)


# ── Codex CLI (TOML file) ────────────────────────────────────────────────────

def _codex_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def _codex_detect() -> bool:
    return _codex_path().parent.exists() or shutil.which("codex") is not None


def _codex_install(server: dict) -> InstallResult:
    """Merge brain into [mcp_servers.brain] in ~/.codex/config.toml."""
    path = _codex_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            return InstallResult(False, f"existing config.toml is invalid TOML: {e}")
        shutil.copy2(path, path.with_suffix(".toml.bak"))
    else:
        data = {}
    servers = data.setdefault("mcp_servers", {})
    if not isinstance(servers, dict):
        return InstallResult(False, "existing mcp_servers is not a table")
    entry: dict = {"command": server["command"]}
    if server.get("args"):
        entry["args"] = server["args"]
    if server.get("env"):
        entry["env"] = server["env"]
    servers["brain"] = entry
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    tmp.replace(path)
    return InstallResult(True, f"wrote {path}")


CLIENTS: tuple[Client, ...] = (
    Client("claude-code",    "Claude Code",    _claude_code_detect,    _claude_code_install),
    Client("claude-desktop", "Claude Desktop", _claude_desktop_detect, _claude_desktop_install, "restart Claude Desktop"),
    Client("cursor",         "Cursor",         _cursor_detect,         _cursor_install,         "restart Cursor"),
    Client("codex",          "Codex",          _codex_detect,          _codex_install,          "restart Codex"),
)


def detect_installed() -> list[Client]:
    return [c for c in CLIENTS if c.detect()]
