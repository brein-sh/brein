"""`brein` CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from . import doctor, mcp_snippet, setup
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
        print(mcp_snippet.snippet(cfg, args.client))
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
    m.set_defaults(func=_cmd_mcp)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
