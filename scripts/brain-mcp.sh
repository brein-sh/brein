#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
export BRAIN_REPO="${BRAIN_REPO:-$HOME/.braincorp/brain}"
exec uv --directory "$ROOT" run brain-mcp
