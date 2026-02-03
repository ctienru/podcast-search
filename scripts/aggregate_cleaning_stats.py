#!/usr/bin/env python3
"""Aggregate cleaning statistics from all cleaned episode files."""

import json
from pathlib import Path
from collections import defaultdict
import sys

def main():
    cleaned_dir = Path(__file__).parent.parent / "data" / "cleaned" / "episodes"

    if not cleaned_dir.exists():
        print(f"Error: {cleaned_dir} does not exist")
        sys.exit(1)

    # Aggregate stats
    total_episodes = 0
    total_paragraphs = 0
    kept_paragraphs = 0
    removed_paragraphs = 0
    total_raw_chars = 0
    total_kept_chars = 0
    removal_breakdown = defaultdict(int)

    # Track episodes with cleaning
    episodes_with_cleaning = 0

    # Process all files
    files = list(cleaned_dir.glob("*.json"))
    print(f"Processing {len(files)} files...")

    for i, f in enumerate(files):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i}/{len(files)}...")

        try:
            data = json.loads(f.read_text())
            stats = data.get("cleaned", {}).get("stats", {})

            total_episodes += 1
            total_paragraphs += stats.get("total_paragraphs", 0)
            kept_paragraphs += stats.get("kept_paragraphs", 0)
            removed_paragraphs += stats.get("removed_paragraphs", 0)
            total_raw_chars += stats.get("raw_char_count", 0)
            total_kept_chars += stats.get("kept_char_count", 0)

            # Aggregate removal breakdown
            breakdown = stats.get("removal_breakdown", {})
            if breakdown:
                episodes_with_cleaning += 1
                for reason, count in breakdown.items():
                    removal_breakdown[reason] += count

        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Calculate percentages
    removal_rate = (removed_paragraphs / total_paragraphs * 100) if total_paragraphs > 0 else 0
    char_reduction = ((total_raw_chars - total_kept_chars) / total_raw_chars * 100) if total_raw_chars > 0 else 0

    # Print results
    print("\n" + "="*60)
    print("CLEANING STATISTICS SUMMARY")
    print("="*60)

    print(f"\n📊 Overall Stats:")
    print(f"  Total episodes processed: {total_episodes:,}")
    print(f"  Episodes with cleaning: {episodes_with_cleaning:,} ({episodes_with_cleaning/total_episodes*100:.1f}%)")

    print(f"\n📝 Paragraph Stats:")
    print(f"  Total paragraphs: {total_paragraphs:,}")
    print(f"  Kept paragraphs: {kept_paragraphs:,}")
    print(f"  Removed paragraphs: {removed_paragraphs:,}")
    print(f"  Removal rate: {removal_rate:.2f}%")

    print(f"\n📏 Character Stats:")
    print(f"  Total raw characters: {total_raw_chars:,}")
    print(f"  Total kept characters: {total_kept_chars:,}")
    print(f"  Characters removed: {total_raw_chars - total_kept_chars:,}")
    print(f"  Character reduction: {char_reduction:.2f}%")

    print(f"\n🔍 Removal Breakdown:")
    total_removals = sum(removal_breakdown.values())
    for reason, count in sorted(removal_breakdown.items(), key=lambda x: -x[1]):
        pct = (count / total_removals * 100) if total_removals > 0 else 0
        print(f"  {reason}: {count:,} ({pct:.1f}%)")

    # Save to JSON
    output = {
        "total_episodes": total_episodes,
        "episodes_with_cleaning": episodes_with_cleaning,
        "total_paragraphs": total_paragraphs,
        "kept_paragraphs": kept_paragraphs,
        "removed_paragraphs": removed_paragraphs,
        "removal_rate_pct": round(removal_rate, 2),
        "total_raw_chars": total_raw_chars,
        "total_kept_chars": total_kept_chars,
        "char_reduction_pct": round(char_reduction, 2),
        "removal_breakdown": dict(removal_breakdown),
    }

    output_path = Path(__file__).parent.parent / "data" / "evaluation" / "cleaning_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n✅ Stats saved to: {output_path}")

if __name__ == "__main__":
    main()
