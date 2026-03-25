"""
Index Health Report

Checks data quality across all language-split Elasticsearch indices.

Checks:
  - Doc count per index
  - Embedding coverage (threshold: >= 99%)
  - show.image_url coverage (threshold: >= 90%)
  - Title / description missing rate
  - Zero-result rate using test queries (threshold: < 5%)

Usage:
    python scripts/index_health_report.py
    python scripts/index_health_report.py --output data/evaluation/index_health.json
    python scripts/index_health_report.py --queries data/evaluation/test_queries.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings
from src.es.client import get_es_client


INDICES = {
    "zh-tw": settings.INDEX_ZH_TW,
    "zh-cn": settings.INDEX_ZH_CN,
    "en": settings.INDEX_EN,
}

THRESHOLDS = {
    "embedding_coverage": 0.99,
    "image_url_coverage": 0.90,
    "zero_result_rate": 0.05,
}


def check_field_coverage(es, index: str, field: str, sample_size: int = 10000) -> float:
    """Return fraction of docs where field exists and is non-null."""
    total_resp = es.count(index=index)
    total = total_resp["count"]
    if total == 0:
        return 0.0

    resp = es.count(
        index=index,
        body={"query": {"exists": {"field": field}}},
    )
    present = resp["count"]
    return present / total


def check_zero_result_rate(es, index: str, queries: List[str]) -> Dict[str, Any]:
    """Run queries against index and return zero-result rate."""
    if not queries:
        return {"rate": None, "zero_result_queries": [], "total_tested": 0}

    zero_result = []
    for q in queries:
        resp = es.search(
            index=index,
            body={
                "query": {
                    "multi_match": {
                        "query": q,
                        "fields": ["title", "title.chinese", "description", "description.chinese"],
                    }
                },
                "size": 1,
                "_source": False,
            },
        )
        if resp["hits"]["total"]["value"] == 0:
            zero_result.append(q)

    return {
        "rate": len(zero_result) / len(queries),
        "zero_result_queries": zero_result,
        "total_tested": len(queries),
    }


def check_index(es, language: str, index_name: str, queries: List[str]) -> Dict[str, Any]:
    """Run all health checks for a single index."""
    print(f"  Checking {index_name}...")

    total_resp = es.count(index=index_name)
    total = total_resp["count"]

    embedding_coverage = check_field_coverage(es, index_name, "embedding")
    image_url_coverage = check_field_coverage(es, index_name, "show.image_url")
    title_coverage = check_field_coverage(es, index_name, "title")
    description_coverage = check_field_coverage(es, index_name, "description")

    zero_result = check_zero_result_rate(es, index_name, queries)

    def gate(value, threshold, lower_is_better=False):
        if value is None:
            return "N/A"
        return "PASS" if (value <= threshold if lower_is_better else value >= threshold) else "FAIL"

    return {
        "index": index_name,
        "language": language,
        "doc_count": total,
        "embedding_coverage": round(embedding_coverage, 4),
        "embedding_coverage_gate": gate(embedding_coverage, THRESHOLDS["embedding_coverage"]),
        "image_url_coverage": round(image_url_coverage, 4),
        "image_url_coverage_gate": gate(image_url_coverage, THRESHOLDS["image_url_coverage"]),
        "title_coverage": round(title_coverage, 4),
        "description_coverage": round(description_coverage, 4),
        "zero_result_rate": round(zero_result["rate"], 4) if zero_result["rate"] is not None else None,
        "zero_result_gate": gate(zero_result["rate"], THRESHOLDS["zero_result_rate"], lower_is_better=True),
        "zero_result_queries": zero_result["zero_result_queries"],
        "zero_result_queries_tested": zero_result["total_tested"],
    }


def load_queries_by_language(path: Path) -> Dict[str, List[str]]:
    """Load test queries grouped by language."""
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    by_language: Dict[str, List[str]] = {"zh-tw": [], "zh-cn": [], "en": []}
    for q in queries:
        lang = q.get("language", "zh-tw")
        if lang == "mixed":
            lang = "zh-tw"
        if lang in by_language:
            by_language[lang].append(q["query"])
    return by_language


def print_report(results: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("Index Health Report")
    print("=" * 60)

    overall = results["overall_gate"]
    print(f"Overall: {'✅ PASS' if overall == 'PASS' else '❌ FAIL'}")
    print()

    for lang, data in results["indices"].items():
        print(f"  [{lang}] {data['index']}  ({data['doc_count']:,} docs)")
        print(f"    Embedding coverage : {data['embedding_coverage']:.1%}  [{data['embedding_coverage_gate']}]  (threshold ≥ 99%)")
        print(f"    Image URL coverage : {data['image_url_coverage']:.1%}  [{data['image_url_coverage_gate']}]  (threshold ≥ 90%)")
        print(f"    Title coverage     : {data['title_coverage']:.1%}")
        print(f"    Description coverage: {data['description_coverage']:.1%}")
        if data["zero_result_rate"] is not None:
            print(f"    Zero-result rate   : {data['zero_result_rate']:.1%}  [{data['zero_result_gate']}]  (threshold < 5%, tested {data['zero_result_queries_tested']} queries)")
            if data["zero_result_queries"]:
                print(f"    Zero-result queries: {data['zero_result_queries']}")
        else:
            print(f"    Zero-result rate   : N/A (no queries provided)")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Index health report for all language indices")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/test_queries.json"),
        help="Path to test queries JSON (used for zero-result rate check)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for JSON report",
    )
    args = parser.parse_args()

    es = get_es_client()

    queries_by_language: Dict[str, List[str]] = {"zh-tw": [], "zh-cn": [], "en": []}
    if args.queries.exists():
        queries_by_language = load_queries_by_language(args.queries)
        print(f"Loaded test queries: zh-tw={len(queries_by_language['zh-tw'])}, zh-cn={len(queries_by_language['zh-cn'])}, en={len(queries_by_language['en'])}")
    else:
        print(f"No query file found at {args.queries}, skipping zero-result check.")

    print()
    index_results: Dict[str, Any] = {}
    for lang, index_name in INDICES.items():
        index_results[lang] = check_index(es, lang, index_name, queries_by_language[lang])

    # Determine overall gate
    gate_checks = ["embedding_coverage_gate", "image_url_coverage_gate", "zero_result_gate"]
    all_pass = all(
        index_results[lang][check] == "PASS"
        for lang in index_results
        for check in gate_checks
        if index_results[lang][check] != "N/A"
    )

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thresholds": THRESHOLDS,
        },
        "overall_gate": "PASS" if all_pass else "FAIL",
        "indices": index_results,
    }

    print_report(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
