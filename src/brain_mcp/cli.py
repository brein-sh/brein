"""`brein` CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import _hooks, consistency, doctor, evolve, index_state, index_worker, mcp_snippet, setup
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

    ev = sub.add_parser("eval", help="Continuous A/B eval (LLM-gated)")
    ev.add_argument(
        "eval_action", choices=["tick", "observe", "capture-prompt"],
        help=("tick: read job JSON from stdin, gate, conditionally run A/B. "
              "observe: PostToolUse hook — fires eval when Grep/Read targets "
              "$BRAIN_REPO. "
              "capture-prompt: UserPromptSubmit hook — extracts `.prompt` "
              "from the Claude Code envelope and writes it to --out."),
    )
    ev.add_argument("--prompt-file", default="",
                    help="Path to the saved user prompt (for `observe`)")
    ev.add_argument("--out", default="",
                    help="Output file (for `capture-prompt`)")
    ev.set_defaults(func=_cmd_eval)

    evo = sub.add_parser(
        "evolve",
        help="Self-improvement: amend brain docs from recent A/B losses",
    )
    evo.add_argument("action", choices=["run", "status"], nargs="?", default="run")
    evo.add_argument("--limit", type=int, default=50,
                     help="Max recent no-brain wins to examine in one run")
    evo.add_argument("--quiet", action="store_true",
                     help="Suppress non-JSON output (used by detached spawn)")
    evo.set_defaults(func=_cmd_evolve)

    dm = sub.add_parser(
        "daemon",
        help="Run a shared HTTP MCP daemon (one model load, many clients)",
    )
    dm.add_argument("action", choices=["run", "url", "launchd"], nargs="?", default="run")
    dm.add_argument("--host", default="127.0.0.1")
    dm.add_argument("--port", type=int, default=8765)
    dm.set_defaults(func=_cmd_daemon)

    return p


def _cmd_eval(args: argparse.Namespace) -> int:
    """`brein eval tick` — run by the detached eval worker. Reads a JSON job
    from stdin: {question, evidence_block, query_hash, trigger}, then runs
    the LLM gate, and if the gate says yes, the full A/B comparison.

    `brein eval observe --prompt-file PATH` — Claude Code PostToolUse hook
    entry. Receives the tool's input JSON on stdin; if the tool targeted a
    path under $BRAIN_REPO, spawns a detached eval worker using the saved
    user prompt as the question. Lets us benchmark grep/read of the brain
    repo as if they were brain_search calls.
    """
    import json
    from . import eval as _eval

    if args.eval_action == "tick":
        try:
            payload = json.loads(sys.stdin.read())
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"eval tick: invalid stdin JSON: {exc}", file=sys.stderr)
            return 2
        _eval._tick(
            question=payload.get("question", ""),
            evidence_block=payload.get("evidence_block", ""),
            query_hash=payload.get("query_hash") or _eval._hash(payload.get("question", "")),
            trigger=payload.get("trigger", "manual"),
        )
        return 0

    if args.eval_action == "capture-prompt":
        try:
            raw = sys.stdin.read()
            text = raw
            try:
                d = json.loads(raw)
                if isinstance(d, dict) and isinstance(d.get("prompt"), str):
                    text = d["prompt"]
            except (json.JSONDecodeError, ValueError):
                pass
            if args.out:
                Path(args.out).write_text(text or "", encoding="utf-8")
        except Exception:
            pass
        return 0

    if args.eval_action == "observe":
        # Silent on every failure — never block Claude Code's tool pipeline.
        try:
            from .config import REPO_PATH
            # BRAIN_OBSERVE_PATHS lets you watch additional brain repos
            # (colon-separated), e.g. legacy or mirrored clones. Defaults to
            # just the primary BRAIN_REPO.
            extra_raw = os.environ.get("BRAIN_OBSERVE_PATHS", "") or ""
            roots: list[Path] = [REPO_PATH]
            for p in extra_raw.split(":"):
                p = p.strip()
                if not p:
                    continue
                try:
                    roots.append(Path(p).expanduser().resolve())
                except (OSError, RuntimeError):
                    pass

            raw = sys.stdin.read() or "{}"
            try:
                envelope = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return 0
            tool_input = (
                envelope.get("tool_input")
                or envelope.get("input")
                or envelope
            )
            path = (
                tool_input.get("file_path")
                or tool_input.get("path")
                or tool_input.get("notebook_path")
                or ""
            )
            if not path:
                return 0
            try:
                target = Path(path).expanduser().resolve()
            except (OSError, RuntimeError):
                return 0

            def _under(t: Path, r: Path) -> bool:
                try:
                    t.relative_to(r)
                    return True
                except ValueError:
                    return False

            if not any(_under(target, r) for r in roots):
                return 0
            prompt_path = Path(args.prompt_file or "")
            if not prompt_path or not prompt_path.exists():
                return 0
            try:
                question = prompt_path.read_text(encoding="utf-8").strip()
            except OSError:
                return 0
            if not question:
                return 0
            if not _eval.EVAL_ENABLED or _eval.EVAL_GUARDED:
                return 0
            if not _eval._which_cli() and not _eval._OR_KEY:
                return 0
            query_hash = _eval._hash(question)
            if _eval._seen_recently(query_hash):
                return 0
            _eval._spawn_eval_worker(
                question=question,
                evidence_block=f"(observed via tool: {path})",
                query_hash=query_hash,
                trigger="tool_observe",
            )
        except Exception:
            pass
        return 0

    return 2


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

    # Pull brain repo/log/index/model/eval flags from the user's config so the
    # daemon has everything it needs at boot. Without these, the daemon would
    # default BRAIN_REPO to ~/.brein/brain (likely non-existent) and silently
    # fail every search.
    try:
        cfg = load()
        cfg_env: list[tuple[str, str]] = [
            ("BRAIN_REPO", cfg.repo_path),
            ("BRAIN_RETRIEVAL_LOG", cfg.retrieval_log),
            ("BRAIN_VECTOR_INDEX", cfg.vector_index),
            ("BRAIN_EMBEDDING_MODEL", cfg.embedding_model),
            ("BRAIN_EVAL_ENABLED", "on" if cfg.eval_enabled else "off"),
        ]
    except Exception:
        cfg_env = []
    cfg_lines = "\n".join(
        f"    <key>{k}</key><string>{v}</string>"
        for k, v in cfg_env if v
    )
    cfg_block = f"\n{cfg_lines}" if cfg_lines else ""

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
    <key>BRAIN_MCP_PORT</key><string>{port}</string>{cfg_block}
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def _cmd_evolve(args: argparse.Namespace) -> int:
    """`brein evolve run` — read recent no-brain wins from eval-log.jsonl,
    invoke the agentic improver on each, commit + push patches.

    `brein evolve status` — print the last 10 evolve runs."""
    if args.action == "status":
        rows = evolve.read_log(limit=10)
        print(json.dumps(rows, indent=2))
        return 0
    result = evolve.run_evolve(limit=args.limit)
    if not args.quiet:
        print(json.dumps(result.to_json(), indent=2))
    return 0


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
