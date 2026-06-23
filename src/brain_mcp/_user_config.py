"""User-facing config file at ~/.brein/config.json.

Server runtime still reads env vars (see config.py). This file is the source
of truth for the setup wizard and the MCP snippet generator — env vars get
populated from here.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("BREIN_HOME", str(Path.home() / ".brein")))
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class BreinConfig:
    repo_path: str = ""
    retrieval_log: str = ""
    vector_index: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    eval_enabled: bool = False
    eval_host_order: list[str] = field(default_factory=lambda: ["claude", "codex", "gemini"])

    def as_env(self) -> dict[str, str]:
        env = {"BRAIN_REPO": self.repo_path}
        if self.retrieval_log:
            env["BRAIN_RETRIEVAL_LOG"] = self.retrieval_log
        if self.vector_index:
            env["BRAIN_VECTOR_INDEX"] = self.vector_index
        if self.embedding_model:
            env["BRAIN_EMBEDDING_MODEL"] = self.embedding_model
        if self.eval_enabled:
            env["BRAIN_EVAL_ENABLED"] = "1"
        return env


def load() -> BreinConfig:
    if not CONFIG_PATH.exists():
        return BreinConfig()
    data = json.loads(CONFIG_PATH.read_text())
    return BreinConfig(**{k: v for k, v in data.items() if k in BreinConfig.__dataclass_fields__})


def save(cfg: BreinConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, CONFIG_PATH.with_suffix(".json.bak"))
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
