"""Bug-hunting tests for the brain-mcp streamable-HTTP transport.

All tests target the HTTP daemon and look for parity bugs vs. stdio,
crash regressions on tool errors, or per-client state leaks.
"""
from __future__ import annotations

import asyncio
import json
import random
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from conftest import call_tool as stdio_call_tool
from conftest import make_frontmatter


BRAIN_MCP_ARGS = [sys.executable, "-m", "brain_mcp.server"]


def _free_port() -> int:
    return random.randint(8800, 9800)


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"daemon never opened {host}:{port}")


@contextmanager
def http_daemon(env: dict[str, str]):
    """Launch brain-mcp in streamable-http mode, yield (host, port), cleanup."""
    port = _free_port()
    host = "127.0.0.1"
    env = {**env,
           "BRAIN_MCP_TRANSPORT": "http",
           "BRAIN_MCP_HOST": host,
           "BRAIN_MCP_PORT": str(port)}
    proc = subprocess.Popen(
        BRAIN_MCP_ARGS, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        try:
            _wait_for_port(host, port)
        except RuntimeError:
            # Daemon failed to bind. Surface its stderr.
            proc.terminate()
            try:
                _, err = proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, err = proc.communicate()
            raise RuntimeError(
                f"daemon failed to start. stderr:\n{err.decode(errors='replace')}"
            )
        yield host, port, proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


async def _http_call_async(host: str, port: int, tool: str, args: dict):
    url = f"http://{host}:{port}/mcp"
    async with streamablehttp_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            assert result.content, f"{tool} returned no content"
            return result.content[0].text, bool(getattr(result, "isError", False))


def http_call(host: str, port: int, tool: str, args: dict) -> tuple[str, bool]:
    return asyncio.run(_http_call_async(host, port, tool, args))


def _parse(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ────────────────────────────────────────────────────────────────────────────
# 1. brain_update parity: HTTP must do the full write→commit→push round-trip
# ────────────────────────────────────────────────────────────────────────────

def test_brain_update_via_http_writes_commits_pushes(brain_env):
    repo = Path(brain_env["BRAIN_REPO"])
    rel = "docs/http_gamma.md"
    content = (
        make_frontmatter("HTTP gamma penguin", ["bird"])
        + "Penguins waddle over HTTP.\n"
    )
    with http_daemon(brain_env) as (host, port, _proc):
        text, is_error = http_call(host, port, "brain_update", {
            "file_path": rel,
            "content": content,
            "commit_message": "test: http gamma penguin",
        })
    assert not is_error, f"brain_update errored over HTTP: {text}"
    out = _parse(text)
    assert isinstance(out, dict), f"expected dict, got {type(out).__name__}: {out!r}"
    assert "error" not in out, out
    assert (repo / rel).read_text() == content, "file did not land on disk"

    log = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--pretty=%s"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert "http gamma penguin" in log, log

    bare_log = subprocess.run(
        ["git", "-C", str(repo.parent / "brain.git"), "log", "-1",
         "--pretty=%s", "refs/heads/main"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert "http gamma penguin" in bare_log, bare_log


# ────────────────────────────────────────────────────────────────────────────
# 2. Parity: identical JSON shape stdio vs. http (brain_search)
# ────────────────────────────────────────────────────────────────────────────

def _shape(obj: Any) -> Any:
    """Return a key-structure skeleton (types only, no leaf values)."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        # collapse to one element's shape (or [] if empty) to allow length differences
        return [_shape(obj[0])] if obj else []
    return type(obj).__name__


def test_brain_search_shape_parity_stdio_vs_http(brain_env):
    args = {"query": "quokka nasturtium"}
    stdio_text, stdio_err = stdio_call_tool(brain_env, "brain_search", args)
    with http_daemon(brain_env) as (host, port, _proc):
        http_text, http_err = http_call(host, port, "brain_search", args)
    assert stdio_err == http_err, (
        f"isError parity broken: stdio={stdio_err} http={http_err}\n"
        f"stdio={stdio_text!r}\nhttp={http_text!r}"
    )
    stdio_obj = _parse(stdio_text)
    http_obj = _parse(http_text)
    assert type(stdio_obj) is type(http_obj), (
        f"top-level type differs: stdio={type(stdio_obj).__name__} "
        f"http={type(http_obj).__name__}"
    )
    assert _shape(stdio_obj) == _shape(http_obj), (
        f"shape mismatch:\nstdio={_shape(stdio_obj)}\nhttp={_shape(http_obj)}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 3. Parity: error envelope shape for invalid file_path
# ────────────────────────────────────────────────────────────────────────────

def test_brain_update_error_envelope_parity(brain_env):
    bad_args = {
        "file_path": "../escape.md",   # outside docs/, should be rejected
        "content": "x",
        "commit_message": "should fail",
    }
    stdio_text, stdio_err = stdio_call_tool(brain_env, "brain_update", bad_args)
    with http_daemon(brain_env) as (host, port, _proc):
        http_text, http_err = http_call(host, port, "brain_update", bad_args)

    assert stdio_err == http_err, (
        f"isError parity broken for invalid path: "
        f"stdio={stdio_err} http={http_err}\n"
        f"stdio={stdio_text!r}\nhttp={http_text!r}"
    )
    stdio_obj = _parse(stdio_text)
    http_obj = _parse(http_text)
    assert _shape(stdio_obj) == _shape(http_obj), (
        f"error envelope shape diverges:\n"
        f"stdio={_shape(stdio_obj)}\nhttp={_shape(http_obj)}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 4. Daemon survives tool errors (does not crash on bad file_path)
# ────────────────────────────────────────────────────────────────────────────

def test_daemon_survives_invalid_file_path(brain_env):
    with http_daemon(brain_env) as (host, port, proc):
        # Trigger an error.
        _t1, _e1 = http_call(host, port, "brain_update", {
            "file_path": "../escape.md",
            "content": "x",
            "commit_message": "boom",
        })
        # Daemon must still be alive and serving.
        assert proc.poll() is None, (
            f"daemon crashed after invalid file_path. "
            f"returncode={proc.returncode}"
        )
        text, is_error = http_call(host, port, "brain_search",
                                   {"query": "quokka"})
        assert not is_error, f"daemon not serving after error: {text}"
        out = _parse(text)
        assert isinstance(out, dict), f"unexpected payload type: {out!r}"


# ────────────────────────────────────────────────────────────────────────────
# 5. Concurrent clients on same daemon — no crash, both get responses
# ────────────────────────────────────────────────────────────────────────────

def test_two_clients_concurrent_on_same_daemon(brain_env):
    with http_daemon(brain_env) as (host, port, proc):
        async def two_clients():
            return await asyncio.gather(
                _http_call_async(host, port, "brain_search",
                                 {"query": "quokka nasturtium"}),
                _http_call_async(host, port, "brain_search",
                                 {"query": "walrus pickles"}),
            )
        results = asyncio.run(two_clients())
        assert proc.poll() is None, "daemon crashed under concurrent clients"

    for text, is_error in results:
        assert not is_error, f"concurrent call errored: {text}"
        out = _parse(text)
        assert isinstance(out, dict), f"unexpected payload: {out!r}"


# ────────────────────────────────────────────────────────────────────────────
# 6. Retrieval-log isolation: client A's writes vs. client B's view
#
# This is a STATE-LEAK probe. With one daemon, BRAIN_RETRIEVAL_LOG is process-
# wide — so two clients SHOULD both see each other's log lines, because there
# is no per-client isolation. The test asserts the daemon at least does not
# duplicate or lose lines under interleaved calls.
# ────────────────────────────────────────────────────────────────────────────

def test_retrieval_log_records_every_http_call(brain_env):
    log_path = Path(brain_env["BRAIN_RETRIEVAL_LOG"])
    before = log_path.read_text().splitlines() if log_path.exists() else []

    queries = [
        "quokka nasturtium",
        "walrus pickles",
        "quokka leaves",
        "walrus moonlight",
    ]
    with http_daemon(brain_env) as (host, port, _proc):
        async def fan_out():
            return await asyncio.gather(*[
                _http_call_async(host, port, "brain_search", {"query": q})
                for q in queries
            ])
        results = asyncio.run(fan_out())

    for text, is_error in results:
        assert not is_error, f"search errored: {text}"

    after = log_path.read_text().splitlines() if log_path.exists() else []
    new_lines = after[len(before):]
    # Every line is valid JSON.
    for ln in new_lines:
        try:
            json.loads(ln)
        except json.JSONDecodeError as e:
            pytest.fail(f"retrieval-log line not JSON: {ln!r} ({e})")

    # Count tool_call rows for brain_search across the new tail.
    tool_calls = [
        json.loads(ln) for ln in new_lines
        if json.loads(ln).get("gen_ai.tool.name") == "brain_search"
        and "gen_ai.operation.name" in json.loads(ln)
    ]
    # stdio test_e2e proves stdio writes exactly one tool_call per search;
    # HTTP must do the same. Anything else (0, duplicates, missing) is a bug.
    assert len(tool_calls) == len(queries), (
        f"expected {len(queries)} tool_call rows over HTTP, got {len(tool_calls)}. "
        f"new_lines={new_lines}"
    )
