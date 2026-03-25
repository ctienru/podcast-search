"""Check NDCG regression gate.

Reads an ndcg_mrr_report.json produced by evaluate_ndcg_mrr.py and exits
non-zero if any metric falls below the v1 baseline thresholds.

Baselines (v1 measured values):
    zh NDCG@10  >= 0.897   (zh-tw + zh-cn combined)
    en NDCG@10  >= 0.853

Gate rule: any language's best-method NDCG@10 must not drop below its
baseline. The best method is whichever of bm25 / embedding / hybrid
scores highest in the report.

Usage:
    python scripts/check_regression_gate.py --report data/evaluation/ndcg_mrr_report.json

Exit codes:
    0  All thresholds met — safe to merge.
    1  One or more thresholds failed — block the PR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# v1 baseline thresholds by language group.
# Keys must match the language values used in test_queries.json.
BASELINES: dict[str, float] = {
    "zh-tw": 0.897,
    "zh-cn": 0.897,
    "en":    0.853,
}

METHODS = ("bm25", "embedding", "hybrid")


def best_ndcg(lang_stats: dict) -> float:
    """Return the highest NDCG@10 across all methods present in lang_stats."""
    values = [
        lang_stats[m]["ndcg@10"]
        for m in METHODS
        if m in lang_stats
    ]
    return max(values) if values else 0.0


def check(report_path: Path) -> bool:
    """Run the regression gate checks.

    Args:
        report_path: Path to the ndcg_mrr_report.json file.

    Returns:
        True if all checks pass, False if any threshold is violated.
    """
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)

    by_language: dict = report.get("by_language", {})
    all_pass = True

    for lang, threshold in BASELINES.items():
        if lang not in by_language:
            print(f"[SKIP] {lang}: not present in report")
            continue

        score = best_ndcg(by_language[lang])

        if score >= threshold:
            print(f"[PASS] {lang}: NDCG@10={score:.4f} >= {threshold}")
        else:
            print(
                f"[FAIL] {lang}: NDCG@10={score:.4f} < threshold {threshold}"
                f" (drop of {threshold - score:.4f})"
            )
            all_pass = False

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="NDCG regression gate")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/evaluation/ndcg_mrr_report.json"),
        help="Path to ndcg_mrr_report.json",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"[ERROR] Report not found: {args.report}", file=sys.stderr)
        sys.exit(1)

    passed = check(args.report)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
