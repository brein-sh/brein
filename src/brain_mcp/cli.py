"""`brein` CLI entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import _hooks, consistency, doctor, index_state, index_worker, mcp_snippet, setup
from ._user_config import CONFIG_PATH, load


def _cmd_setup(args: argparse.Namespace) -> int:
    return setup.run(section=args.section)


def _cmd_doctor(args: argparse.Namespace) -> int:
    return doctor.run()


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load()
    if args.config_action == "path":
        print(CONFIG_PATH)
    else:
        from dataclasses import asdict
        import json
        print(json.dumps(asdict(cfg), indent=2))
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    cfg = load()
    try:
        print(mcp_snippet.snippet(cfg, args.client, http_url=args.http_url))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brein", description="Brain MCP setup & diagnostics")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="Interactive setup wizard")
    s.add_argument(
        "section", nargs="?",
        choices=[sec.key for sec in setup.SECTIONS],
        help="Run only one section (default: all)",
    )
    s.set_defaults(func=_cmd_setup)

    d = sub.add_parser("doctor", help="Run health checks")
    d.set_defaults(func=_cmd_doctor)

    c = sub.add_parser("config", help="Show config")
    c.add_argument("config_action", nargs="?", default="show", choices=["show", "path"])
    c.set_defaults(func=_cmd_config)

    m = sub.add_parser("mcp", help="Print MCP client snippet")
    m.add_argument("client", choices=mcp_snippet.CLIENTS)
    m.add_argument(
        "--http-url",
        help="Use the shared daemon at this URL (e.g. http://127.0.0.1:8765/mcp) "
             "instead of spawning stdio per client",
    )
    m.set_defaults(func=_cmd_mcp)

    h = sub.add_parser("hooks", help="Manage Claude Code hooks")
    h.add_argument("action", choices=["on", "off", "status", "install"])
    h.set_defaults(func=_cmd_hooks)

    i = sub.add_parser("index", help="Manage the vector index")
    i.add_argument("action", choices=["build", "spawn", "status", "reset"])
    i.set_defaults(func=_cmd_index)

    cc = sub.add_parser("consistency", help="Background consistency checker for brain writes")
    cc.add_argument("action", choices=["check", "spawn", "status", "clear"])
    cc.add_argument("path", nargs="?", help="repo-relative or absolute path to the written doc (for check/spawn)")
    cc.set_defaults(func=_cmd_consistency)

    dm = sub.add_parser(
        "daemon",
        help="Run a shared HTTP MCP daemon (one model load, many clients)",
    )
    dm.add_argument("action", choices=["run", "url", "launchd"], nargs="?", default="run")
    dm.add_argument("--host", default="127.0.0.1")
    dm.add_argument("--port", type=int, default=8765)
    dm.set_defaults(func=_cmd_daemon)

    return p


def _cmd_consistency(args: argparse.Namespace) -> int:
    if args.action in {"check", "spawn"} and not args.path:
        print(f"path is required for `consistency {args.action}`", file=sys.stderr)
        return 2
    if args.action == "check":
        finding = consistency.run_check(args.path)
        if finding is None:
            print("ok — no finding emitted (no nearby docs, or judge said 'ok')")
            return 0
        import json as _json
        print(_json.dumps(finding.to_json(), indent=2))
        return 0
    if args.action == "spawn":
        pid = consistency.spawn_detached(args.path)
        print(f"consistency worker spawned (pid={pid}); tail ~/.brein/consistency-worker.log")
        return 0
    if args.action == "clear":
        n = consistency.clear_queue()
        print(f"cleared {n} findings from queue")
        return 0
    # status
    q = consistency.read_queue()
    print(f"queued findings: {len(q)}")
    for f in q[-10:]:
        print(f"  • [{f.kind}/{f.confidence}] {f.write_path} — {f.summary}")
        if f.related_paths:
            print(f"    related: {', '.join(f.related_paths[:3])}")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    if args.action == "build":
        return index_worker.run()
    if args.action == "spawn":
        pid = index_worker.spawn_detached()
        print(f"index worker spawned (pid={pid}); tail ~/.brein/index-worker.log")
        return 0
    if args.action == "reset":
        index_state.clear()
        print("index state cleared")
        return 0
    # status
    status, state = index_state.resolve_status()
    print(f"status: {status}")
    if state:
        print(f"  started_at: {state.started_at}")
        print(f"  updated_at: {state.updated_at}")
        print(f"  worker_pid: {state.worker_pid}")
        print(f"  progress:   {state.done}/{state.total}")
        if state.last_error:
            print(f"  last_error: {state.last_error.splitlines()[0]}")
    return 0


def _cmd_hooks(args: argparse.Namespace) -> int:
    if args.action == "install":
        try:
            path = _hooks.install()
        except RuntimeError as e:
            print(f"install failed: {e}", file=sys.stderr)
            return 1
        print(f"wrote brein hooks into {path}")
        return 0
    if args.action == "on":
        _hooks.set_enabled(True)
        print("brein hooks: ON")
        return 0
    if args.action == "off":
        _hooks.set_enabled(False)
        print("brein hooks: OFF (run `brein hooks on` to re-enable)")
        return 0
    # status
    s = _hooks.status()
    print(f"installed: {s['installed']}")
    print(f"enabled:   {s['enabled']}")
    return 0


_INSTALL_URL = "git+https://github.com/brein-sh/brein.git"


_LAUNCHD_LABEL = "sh.brein.daemon"


def _launchd_plist(host: str, port: int, brain_mcp_path: str) -> str:
    log = Path.home() / ".brein" / "daemon.log"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{_LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{brain_mcp_path}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>BRAIN_MCP_TRANSPORT</key><string>http</string>
    <key>BRAIN_MCP_HOST</key><string>{host}</string>
    <key>BRAIN_MCP_PORT</key><string>{port}</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def _cmd_daemon(args: argparse.Namespace) -> int:
    import shutil

    url = f"http://{args.host}:{args.port}/mcp"

    if args.action == "url":
        print(url)
        return 0

    if args.action == "launchd":
        brain_mcp_path = shutil.which("brain-mcp") or "/usr/local/bin/brain-mcp"
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCHD_LABEL}.plist"
        print(_launchd_plist(args.host, args.port, brain_mcp_path))
        print(f"# save as: {plist_path}", file=sys.stderr)
        print(f"# load:    launchctl load {plist_path}", file=sys.stderr)
        print(f"# stop:    launchctl unload {plist_path}", file=sys.stderr)
        return 0

    # run: foreground HTTP server. Background it with launchd (`brein daemon
    # launchd`) or `nohup brein daemon > ~/.brein/daemon.log 2>&1 &`.
    os.environ["BRAIN_MCP_TRANSPORT"] = "http"
    os.environ["BRAIN_MCP_HOST"] = args.host
    os.environ["BRAIN_MCP_PORT"] = str(args.port)
    brain_mcp = shutil.which("brain-mcp")
    if brain_mcp:
        os.execvp(brain_mcp, [brain_mcp])
    from . import server
    server.main()
    return 0


def _self_upgrade_and_reexec() -> None:
    """`init` semantics: pull the latest brein from main, then re-exec setup.
    Network failure / missing uv is non-fatal — fall through to in-process setup."""
    import os
    import shutil
    import subprocess

    if not shutil.which("uv"):
        return
    print("Upgrading brein from main…")
    try:
        subprocess.run(
            ["uv", "tool", "install", "--force", "--quiet", _INSTALL_URL],
            check=False, timeout=180,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  upgrade skipped (timeout or uv missing); continuing with current code.", file=sys.stderr)
        return
    # Re-exec from the (now-upgraded) on-disk binary so the new code runs.
    target = shutil.which("brein") or sys.argv[0]
    os.execv(target, [target, "setup", *sys.argv[2:]])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "init":
        _self_upgrade_and_reexec()
        argv[0] = "setup"  # fallback if exec didn't happen
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
