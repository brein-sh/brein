from __future__ import annotations

import datetime
import os
import re
import sys
import urllib.parse
from pathlib import Path

# Bundled with the brein package, but operates on the user's brain repo.
# Resolve the repo root from BRAIN_REPO env (set by the MCP server before
# invocation), falling back to cwd for direct CLI use.
ROOT = Path(os.environ.get("BRAIN_REPO") or os.getcwd()).resolve()
CURATED_DIRS = [ROOT / "docs", ROOT / "skills"]
REQUIRED_DOC_PATTERNS = [
    "title:",
    "owner:",
    "status:",
    "last_reviewed:",
    "review_cycle:",
    "tags:",
    # OKF v0.1: every non-reserved concept doc declares a `type`.
    # See https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
    "type:",
]
# OKF-reserved filenames that do not require a `type` field.
OKF_RESERVED_FILENAMES = {"index.md", "log.md"}
REQUIRED_SKILL_PATTERNS = [
    "name:",
    "description:",
    "version:",
    "author:",
    "license:",
    "metadata:",
]
ALLOWED_DOC_STATUSES = {
    "accepted",
    "active",
    "alias",
    "archived",
    "decided",
    "deprecated",
    "draft",
    "proposed",
    "snapshot",
    "superseded",
}

# review_cycle -> max days since last_reviewed before doc is stale.
REVIEW_CYCLE_DAYS = {
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
    "biannual": 180,
    "annual": 365,
}

FRONTMATTER_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def parse_frontmatter_block(text: str) -> dict[str, str] | None:
    """Extract scalar top-level keys from a --- ... --- frontmatter block."""
    if not text.startswith("---"):
        return None
    rest = text[3:]
    if rest.startswith("\n"):
        rest = rest[1:]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return None
    block = rest[:end_idx]
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line[0] in (" ", "\t"):
            continue
        m = FRONTMATTER_LINE_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def parse_iso_date(value: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(value.strip())
    except (TypeError, ValueError):
        return None


def check_staleness(
    path: Path, fm: dict[str, str], today: datetime.date
) -> list[str]:
    """Return staleness/cycle errors for one doc."""
    errs: list[str] = []
    raw_date = fm.get("last_reviewed", "").strip()
    raw_cycle = fm.get("review_cycle", "").strip().lower()

    last_reviewed = parse_iso_date(raw_date) if raw_date else None
    if raw_date and last_reviewed is None:
        errs.append(
            f"{path}: malformed last_reviewed '{raw_date}' (expected ISO YYYY-MM-DD)"
        )

    if raw_cycle and raw_cycle not in REVIEW_CYCLE_DAYS:
        errs.append(
            f"{path}: unknown review_cycle '{raw_cycle}' "
            f"(expected one of: {', '.join(sorted(REVIEW_CYCLE_DAYS))})"
        )
        return errs

    if last_reviewed is None or not raw_cycle:
        return errs

    cycle_days = REVIEW_CYCLE_DAYS[raw_cycle]
    age = (today - last_reviewed).days
    if age > cycle_days:
        errs.append(
            f"{path}: stale — last_reviewed {raw_date} exceeds {raw_cycle} review cycle "
            f"({age} days > {cycle_days})"
        )
    return errs


def check_status(path: Path, fm: dict[str, str]) -> list[str]:
    """Return status taxonomy errors for one doc."""
    raw_status = fm.get("status", "").strip().lower()
    if not raw_status:
        return []
    if raw_status not in ALLOWED_DOC_STATUSES:
        return [
            f"{path}: unknown status '{raw_status}' "
            f"(expected one of: {', '.join(sorted(ALLOWED_DOC_STATUSES))})"
        ]
    return []


def check_hub_readme_entries() -> list[str]:
    """Ensure temporal hub READMEs enumerate their corresponding records."""
    errs: list[str] = []
    events_dir = ROOT / "docs" / "events"
    events_readme = events_dir / "README.md"
    if events_readme.exists():
        readme_text = events_readme.read_text(encoding="utf-8")
        for path in sorted(events_dir.glob("*.md")):
            if path.name in {"README.md", "template.md"}:
                continue
            if not re.match(r"^\d{4}-\d{2}-\d{2} .+\.md$", path.name):
                continue
            encoded = urllib.parse.quote(path.name)
            if path.name not in readme_text and encoded not in readme_text:
                errs.append(f"{events_readme}: missing event record link for {path.name}")

    calendars_dir = ROOT / "docs" / "calendars"
    calendars_readme = calendars_dir / "README.md"
    if calendars_readme.exists():
        readme_text = calendars_readme.read_text(encoding="utf-8")
        for path in sorted(calendars_dir.glob("*.md")):
            if path.name in {"README.md", "template.md"}:
                continue
            if not re.match(r"^\d{4}-W\d{2}\.md$", path.name):
                continue
            if path.name not in readme_text and path.stem not in readme_text:
                errs.append(f"{calendars_readme}: missing weekly summary link for {path.name}")
    return errs


def main() -> int:
    errors: list[str] = []
    today = datetime.date.today()

    for base in CURATED_DIRS:
        for path in base.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            if path.name == "README.md" and path.parent == ROOT / "skills":
                continue
            if path.name in {
                "README.md",
                "CONTRIBUTING.md",
                "SECURITY.md",
                "PULL_REQUEST_TEMPLATE.md",
            }:
                continue
            if path.is_relative_to(ROOT / "docs"):
                # OKF reserved files (index.md, log.md) don't require frontmatter.
                if path.name in OKF_RESERVED_FILENAMES and not text.startswith("---"):
                    continue
                if not text.startswith("---"):
                    errors.append(f"{path}: missing frontmatter block")
                    continue
                if not re.search(r"\n---\s*\n", text[3:]):
                    errors.append(f"{path}: frontmatter closing --- not found")
                missing_field = False
                is_okf_reserved = path.name in OKF_RESERVED_FILENAMES
                for pattern in REQUIRED_DOC_PATTERNS:
                    # OKF reserved files (index.md, log.md) don't need `type`.
                    if pattern == "type:" and is_okf_reserved:
                        continue
                    if pattern not in text:
                        errors.append(f"{path}: missing {pattern}")
                        missing_field = True
                if not missing_field:
                    fm = parse_frontmatter_block(text) or {}
                    errors.extend(check_status(path, fm))
                    errors.extend(check_staleness(path, fm, today))
            if path.is_relative_to(ROOT / "skills") and path.name == "SKILL.md":
                if not text.startswith("---"):
                    errors.append(f"{path}: skill frontmatter must start at byte 0")
                elif not re.search(r"\n---\s*\n", text[3:]):
                    errors.append(f"{path}: skill frontmatter closing --- not found")
                for pattern in REQUIRED_SKILL_PATTERNS:
                    if pattern not in text:
                        errors.append(f"{path}: missing {pattern}")

    errors.extend(check_hub_readme_entries())

    if errors:
        for err in errors:
            print(err)
        return 1

    print("docs validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
