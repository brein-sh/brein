"""Generate MCP client config snippets from BreinConfig."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from ._user_config import BreinConfig

CLIENTS = ("claude", "cursor", "codex", "generic")


def _launcher_command() -> list[str]:
    """Path to brain-mcp launcher. Prefer the console script; fall back to scripts/brain-mcp.sh."""
    bin_path = shutil.which("brain-mcp")
    if bin_path:
        return [bin_path]
    repo_root = Path(__file__).resolve().parents[2]
    sh = repo_root / "scripts" / "brain-mcp.sh"
    if sh.exists():
        return [str(sh)]
    return [sys.executable, "-m", "brain_mcp.server"]


def snippet(cfg: BreinConfig, client: str) -> str:
    if client not in CLIENTS:
        raise ValueError(f"unknown client {client!r}, expected one of {CLIENTS}")
    if not cfg.repo_path:
        raise ValueError("repo_path is empty — run `brein setup` first")

    cmd = _launcher_command()
    server_block = {
        "command": cmd[0],
        "args": cmd[1:],
        "env": cfg.as_env(),
    }
    if not server_block["args"]:
        server_block.pop("args")

    # All four supported clients use the same `mcpServers` envelope today.
    payload = {"mcpServers": {"brain": server_block}}
    return json.dumps(payload, indent=2)
