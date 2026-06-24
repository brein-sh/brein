"""Claude Code hook entries for brein.

Installed into ~/.claude/settings.json under `hooks`. Each entry carries a
`_brein: true` sentinel so we can find and replace them on re-install
without touching unrelated hooks.

Runtime toggle: any hook short-circuits when ~/.brein/disabled exists.
`brein hooks on/off` flips that file.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
DISABLE_FLAG = Path.home() / ".brein" / "disabled"

_DISABLE_CHECK = '[ "${BREIN_GATE:-on}" = "off" ] && exit 0; [ -f "$HOME/.brein/disabled" ] && exit 0; '
_SEARCH_FLAG = '/tmp/claude-brein-search-${CLAUDE_CODE_SESSION_ID:-default}'
_WRITE_FLAG = '/tmp/claude-brein-write-${CLAUDE_CODE_SESSION_ID:-default}'
_WRITE_REMINDED = '/tmp/claude-brein-write-reminded-${CLAUDE_CODE_SESSION_ID:-default}'


def _entry(matcher: str, command: str) -> dict:
    return {
        "_brein": True,
        "matcher": matcher,
        "hooks": [{"type": "command", "command": command}],
    }


def entries() -> dict[str, list[dict]]:
    """All brein hook entries keyed by Claude Code event type."""
    read_gate = (
        f'{_DISABLE_CHECK}'
        f'F="{_SEARCH_FLAG}"; [ -f "$F" ] && exit 0; '
        "echo '[BLOCKED] Call mcp__brain__brain_search first (or `brein hooks off`).' >&2; "
        "exit 2"
    )
    write_reminder = (
        f'{_DISABLE_CHECK}'
        f'W="{_WRITE_FLAG}"; [ -f "$W" ] && exit 0; '
        f'R="{_WRITE_REMINDED}"; [ -f "$R" ] && exit 0; '
        'touch "$R"; '
        "echo '[REMINDER] No brain writes this session. brain_update durable learnings.' >&2; "
        "exit 0"
    )
    return {
        "PreToolUse": [_entry(r"^(?!ToolSearch$|mcp__brain__).+", read_gate)],
        "PostToolUse": [
            _entry("mcp__brain__brain_search",   f'touch "{_SEARCH_FLAG}"'),
            _entry("mcp__brain__brain_evidence", f'touch "{_SEARCH_FLAG}"'),
            _entry("mcp__brain__brain_update",   f'touch "{_WRITE_FLAG}"'),
        ],
        "Stop": [_entry("", write_reminder)],
    }


def install() -> str:
    """Merge brein hook entries into ~/.claude/settings.json. Idempotent —
    existing `_brein: true` entries are stripped before re-inserting."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"existing settings.json is invalid JSON: {e}")
        shutil.copy2(SETTINGS_PATH, SETTINGS_PATH.with_suffix(".json.bak"))
    else:
        data = {}
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise RuntimeError("settings.json `hooks` is not an object")

    for event, new_entries in entries().items():
        existing = hooks.get(event, [])
        if not isinstance(existing, list):
            existing = []
        # Strip our previous entries; preserve everything else.
        kept = [e for e in existing if not (isinstance(e, dict) and e.get("_brein"))]
        hooks[event] = kept + new_entries

    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(SETTINGS_PATH)
    return str(SETTINGS_PATH)


def status() -> dict:
    enabled = not DISABLE_FLAG.exists()
    installed = False
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
            for event_hooks in data.get("hooks", {}).values():
                if any(isinstance(e, dict) and e.get("_brein") for e in event_hooks or []):
                    installed = True
                    break
        except json.JSONDecodeError:
            pass
    return {"installed": installed, "enabled": enabled}


def set_enabled(enabled: bool) -> None:
    DISABLE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    if enabled:
        DISABLE_FLAG.unlink(missing_ok=True)
    else:
        DISABLE_FLAG.touch()
