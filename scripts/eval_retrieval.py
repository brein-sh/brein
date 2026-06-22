#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Override with --cases CLI flag in practice; fallback to repo-local examples.
EVAL_CASES_PATH = REPO_ROOT / "evals" / "brain_retrieval_eval_cases.json"

# Make `brain_mcp` importable when running this script directly without
# `uv run`. Falls back to the installed package otherwise.
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brain_mcp import server  # noqa: E402


def load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    payload = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if limit is not None:
        cases = cases[:limit]
    return cases


def parse_search(query: str, mode: str, rerank: bool = False, rerank_method: str = "heuristic") -> dict[str, Any]:
    raw = server.brain_search(
        query=query,
        directory="docs",
        max_results=10,
        mode=mode,
        rerank=rerank,
        rerank_top_k=25,
        rerank_method=rerank_method,
    )
    data = json.loads(raw)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data


def score_case(case: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_docs = case.get("retrieval", {}).get("expected_docs", []) or []
    forbidden_docs = case.get("retrieval", {}).get("forbidden_top_docs", []) or []
    top_paths = [str(result.get("path", "")) for result in results]
    top5 = top_paths[:5]
    top10 = top_paths[:10]

    hit_ranks = [i + 1 for i, path in enumerate(top_paths) if path in expected_docs]
    first_hit = min(hit_ranks) if hit_ranks else None

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "priority": case["priority"],
        "expected_docs": expected_docs,
        "forbidden_top_docs": forbidden_docs,
        "top_docs": top_paths,
        "matched_expected_docs": [path for path in top_paths if path in expected_docs],
        "top1_expected": bool(top_paths) and top_paths[0] in expected_docs,
        "recall_at_5": 1.0 if expected_docs and any(path in expected_docs for path in top5) else (None if not expected_docs else 0.0),
        "recall_at_10": 1.0 if expected_docs and any(path in expected_docs for path in top10) else (None if not expected_docs else 0.0),
        "mrr": (1.0 / first_hit) if first_hit else (None if not expected_docs else 0.0),
        "forbidden_hit_top1": bool(top_paths) and top_paths[0] in forbidden_docs,
        "forbidden_hit_top5": any(path in forbidden_docs for path in top5),
        "forbidden_hit_top10": any(path in forbidden_docs for path in top10),
    }


def aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [c for c in case_results if c["expected_docs"]]
    unknown = [c for c in case_results if not c["expected_docs"]]

    def mean(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    recall5 = mean([float(c["recall_at_5"]) for c in scored if c["recall_at_5"] is not None])
    recall10 = mean([float(c["recall_at_10"]) for c in scored if c["recall_at_10"] is not None])
    mrr = mean([float(c["mrr"]) for c in scored if c["mrr"] is not None])
    top1_expected = mean([1.0 if c["top1_expected"] else 0.0 for c in scored])
    forbidden_top1 = sum(1 for c in case_results if c["forbidden_hit_top1"])
    forbidden_top5 = sum(1 for c in case_results if c["forbidden_hit_top5"])
    forbidden_top10 = sum(1 for c in case_results if c["forbidden_hit_top10"])

    misses = [
        {
            "id": c["id"],
            "expected_docs": c["expected_docs"],
            "top_docs": c["top_docs"][:5],
        }
        for c in scored
        if not c["recall_at_10"]
    ]

    return {
        "cases_total": len(case_results),
        "cases_scored": len(scored),
        "cases_unknown": len(unknown),
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "mrr": mrr,
        "top1_expected": top1_expected,
        "forbidden_top1_hits": forbidden_top1,
        "forbidden_top5_hits": forbidden_top5,
        "forbidden_top10_hits": forbidden_top10,
        "misses": misses,
    }


def evaluate_mode(cases: list[dict[str, Any]], mode: str, rerank_method: str, allow_llm: bool) -> dict[str, Any]:
    if mode == "hybrid_rerank" and rerank_method == "llm" and not allow_llm:
        return {
            "mode": mode,
            "available": False,
            "reason": "llm rerank requested but --allow-llm was not set",
            "summary": None,
            "cases": [],
        }

    if mode == "hybrid_rerank":
        search_mode = "hybrid"
        rerank = True
        effective_rerank_method = rerank_method if rerank_method != "auto" else "heuristic"
    else:
        search_mode = mode
        rerank = False
        effective_rerank_method = "heuristic"

    case_results: list[dict[str, Any]] = []
    unavailable_reason = None
    for case in cases:
        try:
            data = parse_search(
                case["question"],
                mode=search_mode,
                rerank=rerank,
                rerank_method=effective_rerank_method,
            )
            results = data.get("results", []) or []
            case_results.append(score_case(case, results))
        except Exception as exc:  # pragma: no cover - surfaced in report
            unavailable_reason = f"{case['id']}: {exc}"
            break

    if unavailable_reason:
        return {
            "mode": mode,
            "available": False,
            "reason": unavailable_reason,
            "summary": None,
            "cases": case_results,
        }

    return {
        "mode": mode,
        "available": True,
        "reason": None,
        "summary": aggregate(case_results),
        "cases": case_results,
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def print_text(report: dict[str, Any]) -> None:
    print(f"brain retrieval evals: {report['cases_total']} cases")
    print(f"source: {EVAL_CASES_PATH}")
    print()
    for mode_name, mode_report in report["modes"].items():
        if not mode_report["available"]:
            print(f"{mode_name}: unavailable ({mode_report['reason']})")
            print()
            continue
        summary = mode_report["summary"]
        print(f"{mode_name}:")
        print(f"  cases scored: {summary['cases_scored']}  unknown: {summary['cases_unknown']}")
        print(f"  recall@5: {format_float(summary['recall_at_5'])}")
        print(f"  recall@10: {format_float(summary['recall_at_10'])}")
        print(f"  mrr: {format_float(summary['mrr'])}")
        print(f"  top1 expected: {format_float(summary['top1_expected'])}")
        print(f"  forbidden top1 hits: {summary['forbidden_top1_hits']}")
        print(f"  forbidden top5 hits: {summary['forbidden_top5_hits']}")
        print(f"  forbidden top10 hits: {summary['forbidden_top10_hits']}")
        if summary["misses"]:
            print("  misses:")
            for miss in summary["misses"][:5]:
                print(f"    - {miss['id']} -> expected {miss['expected_docs']} got {miss['top_docs']}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local brain retrieval evals.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of eval cases to run.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="Modes to run: keyword, vector, hybrid, hybrid_rerank. Default runs all.",
    )
    parser.add_argument(
        "--rerank-method",
        choices=["auto", "heuristic", "llm"],
        default="auto",
        help="Rerank method for hybrid_rerank. Default uses heuristic. LLM is optional and off by default.",
    )
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="Allow llm rerank attempts if the server supports it.",
    )
    args = parser.parse_args()

    cases = load_cases(limit=args.limit)
    mode_names = args.modes or ["keyword", "vector", "hybrid", "hybrid_rerank"]
    report: dict[str, Any] = {
        "cases_total": len(cases),
        "cases_path": str(EVAL_CASES_PATH),
        "modes": {},
    }
    for mode_name in mode_names:
        report["modes"][mode_name] = evaluate_mode(cases, mode_name, args.rerank_method, args.allow_llm)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
