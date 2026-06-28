"""Vector post-rank boost: source_of_truth + recency tiebreaker."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from conftest import brain_env, needs_embedder, run  # noqa: F401


def _write_doc(repo: Path, rel: str, *, title: str, source_of_truth: bool, date: str, body: str) -> None:
    fm = (
        "---\n"
        f"title: {title}\n"
        "owner: tests\n"
        "status: active\n"
        f"last_reviewed: {date}\n"
        "review_cycle: annual\n"
        "tags: [pricing]\n"
        "type: note\n"
        + (f"source_of_truth: {'true' if source_of_truth else 'false'}\n")
        + "---\n\n"
    )
    (repo / rel).write_text(fm + body + "\n")


@needs_embedder
def test_source_of_truth_doc_beats_narrative_with_similar_vector(brain_env):
    """A canonical decision doc and a narrative note both mention the topic.

    Without the boost, vector similarity can equal-rank them or even favour the
    narrative because of richer prose. The boost should land the
    source_of_truth doc at #1.
    """
    repo = Path(brain_env["BRAIN_REPO"])
    # Wipe seed corpus so signal isn't drowned out.
    for seeded in (repo / "docs").glob("*.md"):
        seeded.unlink()
    _write_doc(
        repo, "docs/canonical-pricing.md",
        title="Pricing decision",
        source_of_truth=True,
        date="2026-06-01",
        body="Pricing is set at $49/month for the standard tier. This is the canonical decision.",
    )
    _write_doc(
        repo, "docs/notes-pricing-talk.md",
        title="Notes from the pricing meeting",
        source_of_truth=False,
        date="2024-01-15",
        body="We talked about pricing at $49/month for the standard tier and debated alternatives at length.",
    )
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-am", "rerank corpus"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "brain_mcp.cli", "index", "build"],
        env=brain_env, check=True, capture_output=True,
    )

    out = run(brain_env, "brain_search", {"query": "what is our pricing"})
    paths = [r["path"] for r in out["results"]]
    assert "docs/canonical-pricing.md" in paths
    # Canonical doc should win the head slot after the boost.
    assert paths[0] == "docs/canonical-pricing.md", paths


@needs_embedder
def test_boost_does_not_evict_strong_vector_hit(brain_env):
    """A clearly-different topic should still win its own query — the boost is
    a tiebreaker, not a replacement for vector ranking."""
    repo = Path(brain_env["BRAIN_REPO"])
    for seeded in (repo / "docs").glob("*.md"):
        seeded.unlink()
    _write_doc(
        repo, "docs/pricing.md",
        title="Pricing canonical",
        source_of_truth=True,
        date="2026-06-01",
        body="Pricing canonical document about the $49/month tier.",
    )
    _write_doc(
        repo, "docs/walrus.md",
        title="Walrus dispatch",
        source_of_truth=False,
        date="2024-01-15",
        body="The walrus contemplates pickles by moonlight on the icy shore.",
    )
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-am", "rerank corpus 2"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "brain_mcp.cli", "index", "build"],
        env=brain_env, check=True, capture_output=True,
    )

    out = run(brain_env, "brain_search", {"query": "walrus pickles moonlight"})
    # The source_of_truth pricing doc must NOT outrank the obviously-relevant walrus doc.
    assert out["results"][0]["path"] == "docs/walrus.md", [r["path"] for r in out["results"]]
