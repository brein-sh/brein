#!/usr/bin/env bash
set -euo pipefail
export BRAIN_REPO="${BRAIN_REPO:-$HOME/.braincorp/brain}"
exec uv --directory "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" run python server.py
