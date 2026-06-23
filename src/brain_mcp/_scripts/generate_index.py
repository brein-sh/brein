#!/usr/bin/env python3
"""Generate docs/index.md from the docs/ tree.

Walks docs/**/*.md, reads frontmatter, and emits a domain-first index that
mirrors the current taxonomy. Run with --check to fail when the on-disk
index drifts from generated output.

Stdlib only (Python 3.11+). No external deps.
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import os
import sys
from pathlib import Path
from urllib.parse import quote

# Bundled with the brein package, but operates on the user's brain repo.
# Resolve the repo root from BRAIN_REPO env (set by the MCP server before
# invocation), falling back to cwd for direct CLI use.
ROOT = Path(os.environ.get("BRAIN_REPO") or os.getcwd()).resolve()
DOCS = ROOT / "docs"
INDEX_PATH = DOCS / "index.md"

# Domains that get full enumeration (subfolder grouping inside each).
DOMAINS = ["company", "engineering", "product", "gtm", "ops"]

# Cross-cutting registries / temporal: too large to enumerate inline, link
# to README/index files instead. These keep the curated entry points.
REGISTRY_LINKS = [
    ("contacts/", "contacts/README.md"),
    ("companies/", "companies/README.md"),
]
TEMPORAL_LINKS = [
    ("calendars/", "calendars/README.md"),
    ("events/", "events/README.md"),
]

# Pretty labels for type subfolders within domains.
SUBFOLDER_LABELS = {
    "architecture": "Architecture",
    "decisions": "Decisions",
    "references": "References",
    "projects": "Projects",
    "plans": "Plans",
    "repos": "Repos",
    "waitlists": "Waitlists",
    "operating": "Operating (brain-meta)",
    "archives": "Archives",
}

# Statuses that exclude a doc from the index.
EXCLUDED_STATUS_PREFIXES = ("archived", "deprecated", "superseded")

# README.md and these files are not enumerated even if present.
SKIP_FILENAMES = {"README.md", "index.md"}


def parse_frontmatter(text: str) -> dict | None:
    """Hand-parse a YAML frontmatter block. Returns dict of top-level keys.

    Only handles scalar values and inline lists. Strips surrounding quotes.
    Returns None if no frontmatter block is found.
    """
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
        # only consider top-level keys (no leading whitespace)
        if line[0] in (" ", "\t"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def url_quote(rel_path: str) -> str:
    """Percent-encode path segments but keep '/' separators."""
    return quote(rel_path, safe="/")


def collect_docs() -> list[tuple[Path, dict]]:
    """Return list of (path, frontmatter) for every doc to include."""
    items: list[tuple[Path, dict]] = []
    for path in sorted(DOCS.rglob("*.md")):
        if path.name in SKIP_FILENAMES:
            continue
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if fm is None:
            continue
        status = (fm.get("status") or "").strip().lower()
        if any(status.startswith(p) for p in EXCLUDED_STATUS_PREFIXES):
            continue
        items.append((path, fm))
    return items


def group_by_domain(
    items: list[tuple[Path, dict]],
) -> dict[str, dict[str, list[tuple[Path, dict]]]]:
    """Group items by top-level domain and then by subfolder.

    Files directly under docs/<domain>/ go into key "" (root-of-domain).
    """
    groups: dict[str, dict[str, list[tuple[Path, dict]]]] = {}
    for path, fm in items:
        rel = path.relative_to(DOCS)
        parts = rel.parts
        if len(parts) < 2:
            # Files directly under docs/ — handled separately as "top level".
            continue
        domain = parts[0]
        subfolder = parts[1] if len(parts) >= 3 else ""
        groups.setdefault(domain, {}).setdefault(subfolder, []).append((path, fm))
    return groups


def top_level_items(items: list[tuple[Path, dict]]) -> list[tuple[Path, dict]]:
    return [(p, fm) for p, fm in items if p.parent == DOCS]


def fmt_entry(path: Path, fm: dict) -> str:
    rel = path.relative_to(DOCS).as_posix()
    title = fm.get("title") or path.stem
    return f"- [{title}]({url_quote(rel)})"


def sort_by_title(entries: list[tuple[Path, dict]]) -> list[tuple[Path, dict]]:
    return sorted(entries, key=lambda x: (x[1].get("title") or x[0].stem).lower())


def render_domain_section(
    domain: str, sub_map: dict[str, list[tuple[Path, dict]]]
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {domain}/")
    lines.append("")
    # Files at the root of the domain first.
    root_files = sub_map.get("", [])
    for path, fm in sort_by_title(root_files):
        lines.append(fmt_entry(path, fm))
    # Then each subfolder, ordered to put architecture/decisions early when present.
    preferred_order = [
        "architecture",
        "decisions",
        "references",
        "plans",
        "projects",
        "repos",
        "waitlists",
        "operating",
        "archives",
    ]
    seen: set[str] = set()
    ordered_subs: list[str] = []
    for name in preferred_order:
        if name in sub_map and name != "":
            ordered_subs.append(name)
            seen.add(name)
    for name in sorted(sub_map.keys()):
        if name and name not in seen:
            ordered_subs.append(name)
    for sub in ordered_subs:
        if sub == "":
            continue
        label = SUBFOLDER_LABELS.get(sub, sub.capitalize())
        lines.append(f"- {label}:")
        for path, fm in sort_by_title(sub_map[sub]):
            rel = path.relative_to(DOCS).as_posix()
            title = fm.get("title") or path.stem
            lines.append(f"  - [{title}]({url_quote(rel)})")
    lines.append("")
    return lines


def render_body(items: list[tuple[Path, dict]]) -> str:
    groups = group_by_domain(items)
    lines: list[str] = []
    lines.append(
        "<!-- AUTO-GENERATED by scripts/generate_index.py — do not edit by hand. "
        "Run `python scripts/generate_index.py` to regenerate. -->"
    )
    lines.append("")
    lines.append("# Knowledge Map")
    lines.append("")
    lines.append(
        "Full taxonomy of the company brain. Domain-first at the top, type folders "
        "inside each domain. Cross-cutting registries (contacts, companies) live at "
        "the root. This file is regenerated from frontmatter — to add or rename an "
        "entry, edit the underlying doc, not this index."
    )
    lines.append("")

    # Top level
    lines.append("## Top level")
    lines.append("")
    for path, fm in sort_by_title(top_level_items(items)):
        lines.append(fmt_entry(path, fm))
    lines.append("- Brain-wide changelog: [log.md](log.md)")
    lines.append("- Domains: " + ", ".join(f"[{d}/]({d}/)" for d in DOMAINS))
    lines.append(
        "- Cross-cutting registries: "
        + ", ".join(f"[{label}]({path})" for label, path in REGISTRY_LINKS)
    )
    lines.append(
        "- Temporal: "
        + ", ".join(f"[{label}]({path})" for label, path in TEMPORAL_LINKS)
    )
    lines.append("- Automation: [skills/](skills/README.md), including the [Skill Catalog](skills/skill-catalog.md)")
    lines.append("")

    # Domain sections. DOMAINS sets the preferred order; any additional
    # top-level folders discovered in groups are appended alphabetically so a
    # newly-introduced domain auto-registers without touching this script.
    excluded_top_level = {"skills", "contacts", "companies", "calendars", "events"}
    discovered = sorted(
        d for d in groups.keys() if d not in DOMAINS and d not in excluded_top_level
    )
    for domain in [*DOMAINS, *discovered]:
        if domain not in groups:
            continue
        lines.extend(render_domain_section(domain, groups[domain]))

    # Cross-cutting registries — link only, do not enumerate (too large).
    lines.append("## Cross-cutting registries")
    lines.append("")
    for label, target in REGISTRY_LINKS:
        lines.append(f"- [{label}]({target})")
    lines.append("")

    # Temporal — link only.
    lines.append("## Temporal")
    lines.append("")
    for label, target in TEMPORAL_LINKS:
        lines.append(f"- [{label}]({target})")
    lines.append("")

    # Skills — link to in-repo catalog plus enumerated skill docs under docs/skills.
    lines.append("## Skills")
    lines.append("")
    skills_dir = DOCS / "skills"
    skills_entries: list[tuple[Path, dict]] = [
        (p, fm) for p, fm in items if p.parent == skills_dir
    ]
    for path, fm in sort_by_title(skills_entries):
        lines.append(fmt_entry(path, fm))

    # Enumerate actual SKILL.md files under repo-root skills/. Skills frontmatter
    # uses `name:` and `description:` instead of `title:`. Link format is
    # repo-root-relative `/skills/company/<slug>/SKILL.md` so links don't break
    # when the doc tree is mirrored or moved.
    skills_root = ROOT / "skills"
    skill_files: list[tuple[str, str, str]] = []  # (name, description, link)
    if skills_root.is_dir():
        for skill_path in sorted(skills_root.rglob("SKILL.md")):
            fm = parse_frontmatter(skill_path.read_text(encoding="utf-8")) or {}
            name = fm.get("name") or skill_path.parent.name
            desc = fm.get("description", "").strip()
            rel = skill_path.relative_to(ROOT).as_posix()
            link = "/" + url_quote(rel)
            skill_files.append((name, desc, link))
    skill_files.sort(key=lambda t: t[0].lower())
    for name, desc, link in skill_files:
        if desc:
            lines.append(f"- [{name}]({link}) — {desc}")
        else:
            lines.append(f"- [{name}]({link})")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def extract_existing_frontmatter() -> str:
    """Read the current index.md frontmatter block verbatim.

    If the file doesn't exist or has no frontmatter, emit a minimal default.
    """
    if not INDEX_PATH.exists():
        return _default_frontmatter()
    text = INDEX_PATH.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return _default_frontmatter()
    rest = text[3:]
    if rest.startswith("\n"):
        rest = rest[1:]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return _default_frontmatter()
    today = datetime.date.today().isoformat()
    block_lines: list[str] = []
    updated = False
    for line in rest[:end_idx].splitlines():
        if line.startswith("last_reviewed:"):
            block_lines.append(f"last_reviewed: {today}")
            updated = True
        else:
            block_lines.append(line)
    if not updated:
        block_lines.append(f"last_reviewed: {today}")
    block = "\n".join(block_lines)
    return f"---\n{block}\n---\n"


def _default_frontmatter() -> str:
    today = datetime.date.today().isoformat()
    return (
        "---\n"
        "title: Knowledge Map\n"
        "owner: pmxt-dev/platform\n"
        "status: active\n"
        f"last_reviewed: {today}\n"
        "review_cycle: quarterly\n"
        "tags: [index, navigation]\n"
        "source_of_truth: true\n"
        "---\n"
    )


def render_full_index() -> str:
    items = collect_docs()
    body = render_body(items)
    frontmatter = extract_existing_frontmatter()
    return frontmatter + "\n" + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if docs/index.md differs from generated output.",
    )
    args = parser.parse_args()

    generated = render_full_index()
    if args.check:
        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
        if current != generated:
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                generated.splitlines(keepends=True),
                fromfile="docs/index.md (on disk)",
                tofile="docs/index.md (generated)",
                n=3,
            )
            sys.stdout.writelines(diff)
            print(
                "\ndocs/index.md is out of sync. "
                "Run `python scripts/generate_index.py` to regenerate.",
                file=sys.stderr,
            )
            return 1
        print("docs/index.md is in sync.")
        return 0

    INDEX_PATH.write_text(generated, encoding="utf-8")
    print(f"wrote {INDEX_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
