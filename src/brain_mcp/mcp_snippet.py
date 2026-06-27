"""Generate MCP client config snippets from BreinConfig."""

from __future__ import annotations

import json
import shutil
import sys

from ._user_config import BreinConfig

CLIENTS = ("claude", "cursor", "codex", "generic")


def _launcher_command() -> list[str]:
    """Path to brain-mcp launcher: console script if on PATH, else `python -m`."""
    bin_path = shutil.which("brain-mcp")
    if bin_path:
        return [bin_path]
    return [sys.executable, "-m", "brain_mcp.server"]


def snippet(cfg: BreinConfig, client: str, *, http_url: str | None = None) -> str:
    if client not in CLIENTS:
        raise ValueError(f"unknown client {client!r}, expected one of {CLIENTS}")
    if not cfg.repo_path:
        raise ValueError("repo_path is empty — run `brein setup` first")

    if http_url:
        # Shared daemon: clients connect to one process over HTTP.
        # `type: "http"` is required by Claude Code (newer versions) — without
        # it, the brain server is parsed into config but never shown in /mcp.
        server_block: dict = {"type": "http", "url": http_url}
    else:
        cmd = _launcher_command()
        server_block = {
            "command": cmd[0],
            "args": cmd[1:],
            "env": cfg.as_env(),
        }
        if not server_block["args"]:
            server_block.pop("args")

    payload = {"mcpServers": {"brain": server_block}}
    return json.dumps(payload, indent=2)
