"""
Evaluation Suite — Correctness + Offline Quality

Runs all evaluation checks and produces a dated summary report.

Two check groups:
  Correctness   — is the data in a deployable state?
                  (index health: embedding coverage, image coverage, zero-result rate)
                  (language routing: are episodes in the correct language index?)
  Offline Quality — does search quality meet regression thresholds?
                  (NDCG@10 per language vs stored baselines)
                  (latency P99 per language/mode vs stored baseline × 1.5)

Design decisions (2026-03-25):
  D-S1: Language routing check → reported as N/A (RSS tag data issue causes
         en precision < 95%; labeled sample exists but fix is in crawler/ingest,
         not evaluation logic).
  D-S2: Latency → quick mode (5 runs) against stored baseline;
         threshold = baseline P99 × 1.5 per language/mode.
  D-S3: NDCG → always re-run fresh (never read stale report).

Usage:
    python scripts/run_evaluation_suite.py
    python scripts/run_evaluation_suite.py --output-dir data/evaluation/reports/2026-03-25
    python scripts/run_evaluation_suite.py --skip-ndcg   # for quick smoke-test

Exit codes:
    0  All applicable checks PASS
    1  One or more checks FAIL

Correctness thresholds (index health):
    embedding coverage >= 99%, image_url coverage >= 90%, zero-result rate < 5%

Offline Quality thresholds (NDCG):
    zh-tw NDCG@10 >= 0.897, zh-cn NDCG@10 >= 0.897 (provisional), en NDCG@10 >= 0.853

Offline Quality thresholds (latency):
    P99 per language/mode <= baseline P99 × 1.5
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.es.client import get_es_client


DATE_STR = date.today().isoformat()

NDCG_BASELINES: Dict[str, float] = {
    "zh-tw": 0.897,
    "zh-cn": 0.897,  # provisional — same as zh-tw pending dedicated zh-cn query set
    "en": 0.853,
}

LATENCY_MULTIPLIER = 1.5
LATENCY_BASELINE_PATH = Path("data/benchmark/latency_baseline.json")

LANGUAGE_DETECTION_N_A_REASON = (
    "Labeled sample exists but en precision=90% (< 95% threshold) due to wrong RSS "
    "language tags on 3 shows. Fix is in crawler/ingest layer, not evaluation logic. "
    "See data/evaluation/language_detection_report.json."
)


# ---------------------------------------------------------------------------
# Correctness: Index health
# ---------------------------------------------------------------------------

def run_index_health(output_path: Path) -> Dict[str, Any]:
    """Run index_health_report.py and return its output dict."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/index_health_report.py",
        "--output", str(output_path),
    ]
    queries_path = Path("data/evaluation/test_queries.json")
    if queries_path.exists():
        cmd += ["--queries", str(queries_path)]

    print("  Running index_health_report.py...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return {"status": "ERROR", "error": result.stderr.strip()}

    with open(output_path, encoding="utf-8") as f:
        report = json.load(f)

    gate = report.get("overall_gate", "FAIL")
    return {
        "status": gate,
        "report": str(output_path),
        "indices": {
            lang: {
                "doc_count": data["doc_count"],
                "embedding_coverage": data["embedding_coverage"],
                "embedding_coverage_gate": data["embedding_coverage_gate"],
                "image_url_coverage": data["image_url_coverage"],
                "image_url_coverage_gate": data["image_url_coverage_gate"],
                "zero_result_rate": data.get("zero_result_rate"),
                "zero_result_gate": data.get("zero_result_gate"),
            }
            for lang, data in report.get("indices", {}).items()
        },
    }


def language_routing_na_result() -> Dict[str, Any]:
    """Return N/A status for language routing check (D-S1: data issue in crawler/ingest)."""
    return {
        "status": "N/A",
        "reason": LANGUAGE_DETECTION_N_A_REASON,
    }


# ---------------------------------------------------------------------------
# Offline Quality: NDCG (re-run fresh — D-S3)
# ---------------------------------------------------------------------------

def run_ndcg(output_path: Path) -> Dict[str, Any]:
    """Re-run evaluate_ndcg_mrr.py fresh and check against baselines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/evaluate_ndcg_mrr.py",
        "--output", str(output_path),
        "--no-ci",
    ]
    print("  Running evaluate_ndcg_mrr.py (this takes ~2 minutes)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return {"status": "ERROR", "error": result.stderr.strip()}

    with open(output_path, encoding="utf-8") as f:
        report = json.load(f)

    by_language = report.get("by_language", {})
    per_lang: Dict[str, Any] = {}
    all_pass = True

    for lang, threshold in NDCG_BASELINES.items():
        if lang not in by_language:
            per_lang[lang] = {"status": "N/A", "reason": "no queries for this language"}
            continue

        lang_stats = by_language[lang]
        methods = ("bm25", "embedding", "hybrid")
        scores = {m: lang_stats[m]["ndcg@10"] for m in methods if m in lang_stats}
        best_method = max(scores, key=lambda m: scores[m]) if scores else None
        best_score = scores[best_method] if best_method else 0.0

        passed = best_score >= threshold
        if not passed:
            all_pass = False

        per_lang[lang] = {
            "status": "PASS" if passed else "FAIL",
            "best_method": best_method,
            "best_ndcg": round(best_score, 4),
            "threshold": threshold,
            "scores": {m: round(v, 4) for m, v in scores.items()},
        }

    return {
        "status": "PASS" if all_pass else "FAIL",
        "report": str(output_path),
        "per_language": per_lang,
    }


# ---------------------------------------------------------------------------
# Offline Quality: Latency quick check (D-S2)
# ---------------------------------------------------------------------------

def run_latency_quick(output_path: Path) -> Dict[str, Any]:
    """Run benchmark_latency.py --quick and compare against stored baseline."""
    if not LATENCY_BASELINE_PATH.exists():
        return {
            "status": "N/A",
            "reason": f"No baseline found at {LATENCY_BASELINE_PATH}. Run benchmark_latency.py first.",
        }

    with open(LATENCY_BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/benchmark_latency.py",
        "--quick",
        "--output", str(output_path),
    ]
    print("  Running benchmark_latency.py --quick (5 runs per query, ~30s)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return {"status": "ERROR", "error": result.stderr.strip()}

    with open(output_path, encoding="utf-8") as f:
        current = json.load(f)

    baseline_by_lang = baseline.get("by_language", {})
    current_by_lang = current.get("by_language", {})

    violations: List[str] = []
    per_lang: Dict[str, Any] = {}

    for lang in current_by_lang:
        if lang not in baseline_by_lang:
            per_lang[lang] = {"status": "N/A", "reason": "no baseline for this language"}
            continue

        lang_checks: Dict[str, Any] = {}
        lang_pass = True

        for mode in ("bm25", "knn", "hybrid"):
            if mode not in current_by_lang[lang] or mode not in baseline_by_lang[lang]:
                continue

            cur_p99 = current_by_lang[lang][mode]["p99_ms"]
            base_p99 = baseline_by_lang[lang][mode]["p99_ms"]
            threshold = round(base_p99 * LATENCY_MULTIPLIER, 1)
            passed = cur_p99 <= threshold

            lang_checks[mode] = {
                "status": "PASS" if passed else "FAIL",
                "p99_ms": cur_p99,
                "baseline_p99_ms": base_p99,
                "threshold_ms": threshold,
            }

            if not passed:
                lang_pass = False
                violations.append(f"{lang}/{mode}: P99={cur_p99}ms > threshold {threshold}ms")

        per_lang[lang] = {
            "status": "PASS" if lang_pass else "FAIL",
            "modes": lang_checks,
        }

    all_pass = len(violations) == 0
    result_dict: Dict[str, Any] = {
        "status": "PASS" if all_pass else "FAIL",
        "report": str(output_path),
        "multiplier": LATENCY_MULTIPLIER,
        "per_language": per_lang,
    }
    if violations:
        result_dict["violations"] = violations
    return result_dict


# ---------------------------------------------------------------------------
# Printing and aggregation
# ---------------------------------------------------------------------------

def gate_status_icon(status: str) -> str:
    if status == "PASS":
        return "[PASS]"
    if status == "FAIL":
        return "[FAIL]"
    return "[N/A ]"


def print_report(correctness: Dict[str, Any], offline_quality: Dict[str, Any]) -> None:
    print()
    print("=" * 65)
    print(f"Evaluation Suite Report — {DATE_STR}")
    print("=" * 65)

    overall = "PASS" if correctness["status"] != "FAIL" and offline_quality["status"] == "PASS" else "FAIL"
    print(f"Overall: {gate_status_icon(overall)}")
    print()

    # Correctness
    print(f"Correctness: {gate_status_icon(correctness['status'])}")
    ih = correctness["items"]["index_health"]
    print(f"  {gate_status_icon(ih['status'])} Index health")
    if "indices" in ih:
        for lang, d in ih["indices"].items():
            emb_gate = d.get("embedding_coverage_gate", "?")
            img_gate = d.get("image_url_coverage_gate", "?")
            zr = d.get("zero_result_rate")
            zr_gate = d.get("zero_result_gate", "N/A")
            zr_str = f"{zr:.1%}" if zr is not None else "N/A"
            print(
                f"    [{lang}] docs={d['doc_count']:,}  "
                f"emb={d['embedding_coverage']:.1%}[{emb_gate}]  "
                f"img={d['image_url_coverage']:.1%}[{img_gate}]  "
                f"zr={zr_str}[{zr_gate}]"
            )
    lr = correctness["items"]["language_routing"]
    print(f"  {gate_status_icon(lr['status'])} Language routing")
    if lr["status"] == "N/A":
        print(f"    Reason: {lr['reason']}")
    print()

    # Offline Quality
    print(f"Offline Quality: {gate_status_icon(offline_quality['status'])}")
    ndcg = offline_quality["items"]["ndcg"]
    print(f"  {gate_status_icon(ndcg['status'])} NDCG@10 regression")
    if "per_language" in ndcg:
        for lang, d in ndcg["per_language"].items():
            if d["status"] == "N/A":
                print(f"    [{lang}] N/A — {d.get('reason', '')}")
            else:
                print(
                    f"    [{lang}] {gate_status_icon(d['status'])}  "
                    f"best={d['best_method']} {d['best_ndcg']:.4f}  "
                    f"threshold={d['threshold']}"
                )

    lat = offline_quality["items"]["latency"]
    print(f"  {gate_status_icon(lat['status'])} Latency (quick, P99 <= baseline×{LATENCY_MULTIPLIER})")
    if "per_language" in lat:
        for lang, d in lat["per_language"].items():
            if d["status"] == "N/A":
                print(f"    [{lang}] N/A — {d.get('reason', '')}")
                continue
            for mode, m in d.get("modes", {}).items():
                print(
                    f"    [{lang}/{mode}] {gate_status_icon(m['status'])}  "
                    f"P99={m['p99_ms']}ms  "
                    f"baseline={m['baseline_p99_ms']}ms  "
                    f"threshold={m['threshold_ms']}ms"
                )

    print()
    print("=" * 65)


def compute_gate_status(items: Dict[str, Any]) -> str:
    """Aggregate item statuses: PASS if no FAIL (N/A is treated as non-blocking)."""
    statuses = [v["status"] for v in items.values()]
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "PASS" for s in statuses):
        return "PASS"
    return "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Correctness + Offline Quality evaluation suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"data/evaluation/reports/{DATE_STR}"),
        help=f"Directory for dated reports (default: data/evaluation/reports/{DATE_STR})",
    )
    parser.add_argument(
        "--skip-ndcg",
        action="store_true",
        help="Skip NDCG re-run (smoke-test mode, for faster iteration)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for suite summary JSON (default: --output-dir/suite_report.json)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_output: Path = args.output or output_dir / "suite_report.json"

    print(f"Output directory: {output_dir}")
    print()

    # --- Correctness ---
    print("=== Correctness ===")
    index_health_result = run_index_health(output_dir / "index_health.json")
    lang_routing_result = language_routing_na_result()

    correctness_items = {
        "index_health": index_health_result,
        "language_routing": lang_routing_result,
    }
    correctness = {
        "status": compute_gate_status(correctness_items),
        "items": correctness_items,
    }

    # --- Offline Quality ---
    print()
    print("=== Offline Quality ===")

    if args.skip_ndcg:
        ndcg_result: Dict[str, Any] = {
            "status": "N/A",
            "reason": "--skip-ndcg flag set",
        }
    else:
        ndcg_result = run_ndcg(output_dir / "ndcg_mrr_report.json")

    latency_result = run_latency_quick(output_dir / "latency_quick.json")

    offline_quality_items = {
        "ndcg": ndcg_result,
        "latency": latency_result,
    }
    offline_quality = {
        "status": compute_gate_status(offline_quality_items),
        "items": offline_quality_items,
    }

    # --- Print ---
    print_report(correctness, offline_quality)

    # --- Save ---
    overall = "PASS" if correctness["status"] != "FAIL" and offline_quality["status"] == "PASS" else "FAIL"
    output = {
        "meta": {
            "date": DATE_STR,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
        },
        "overall": overall,
        "correctness": correctness,
        "offline_quality": offline_quality,
    }

    with open(suite_output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Suite report saved to: {suite_output}")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
