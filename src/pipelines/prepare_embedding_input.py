"""Prepare Embedding Input Pipeline

Read cleaned episode JSON files (Layer 2) and produce embedding_input files (Layer 3).
This is a prerequisite for embed_episodes.py, which reads from Layer 3.

Data flow:
    data/cleaned/episodes/*.json  →  data/embedding_input/episodes/*.json

Embedding text format (text-v1):
    "{normalized_title}\\n\\n{kept_paragraphs joined by \\n\\n}"

If there are no kept paragraphs the text is just the title.

Usage:
    python -m src.pipelines.prepare_embedding_input
    python -m src.pipelines.prepare_embedding_input --show-ids id1 id2
    python -m src.pipelines.prepare_embedding_input --force
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

CLEANED_EPISODES_DIR = Path("data/cleaned/episodes")
EMBEDDING_INPUT_DIR = Path("data/embedding_input/episodes")


def _build_embedding_text(cleaned: Dict) -> str:
    """Assemble embedding text from a Layer 2 cleaned episode dict.

    Strategy (text-v1):
    - Title from cleaned.normalized.title
    - Body from paragraphs where kept=True, joined by double newline
    - If no kept paragraphs, returns title only
    """
    normalized = cleaned.get("cleaned", {}).get("normalized", {})
    title = (normalized.get("title") or "").strip()

    paragraphs = cleaned.get("cleaned", {}).get("paragraphs", [])
    kept_texts = [p["text"] for p in paragraphs if p.get("kept") and p.get("text")]

    if kept_texts:
        return title + "\n\n" + "\n\n".join(kept_texts)
    return title


def _process_one(
    cleaned_path: Path,
    output_dir: Path,
    allowed_show_ids: Optional[set],
    force: bool,
) -> str:
    """Process one cleaned episode file.

    Returns one of: "written", "skipped", "filtered", "failed"
    """
    try:
        with open(cleaned_path, encoding="utf-8") as f:
            cleaned = json.load(f)
    except Exception as e:
        logger.warning("cleaned_episode_load_failed", extra={"file": str(cleaned_path), "error": str(e)})
        return "failed"

    episode_id = cleaned.get("episode_id")
    show_id = cleaned.get("show_id")

    if not episode_id or not show_id:
        logger.warning("cleaned_episode_missing_ids", extra={"file": str(cleaned_path)})
        return "failed"

    if allowed_show_ids is not None and show_id not in allowed_show_ids:
        return "filtered"

    safe_name = episode_id.replace(":", "_") + ".json"
    out_path = output_dir / safe_name

    if not force and out_path.exists():
        return "skipped"

    text = _build_embedding_text(cleaned)
    payload = {
        "episode_id": episode_id,
        "show_id": show_id,
        "embedding_input": {"text": text},
    }

    try:
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return "written"
    except Exception as e:
        logger.warning("embedding_input_write_failed", extra={"episode_id": episode_id, "error": str(e)})
        return "failed"


def run(
    allowed_show_ids: Optional[set] = None,
    force: bool = False,
    cleaned_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Build embedding_input files from cleaned episode JSON.

    Args:
        allowed_show_ids: Restrict to these show IDs. None = all shows.
        force:            Overwrite existing embedding_input files.
        cleaned_dir:      Source directory. Defaults to CLEANED_EPISODES_DIR.
        output_dir:       Destination directory. Defaults to EMBEDDING_INPUT_DIR.

    Returns:
        Stats dict with keys: written, skipped, failed, total.
    """
    _cleaned_dir = cleaned_dir or CLEANED_EPISODES_DIR
    _output_dir = output_dir or EMBEDDING_INPUT_DIR
    _output_dir.mkdir(parents=True, exist_ok=True)

    if not _cleaned_dir.exists():
        logger.warning("cleaned_episodes_dir_not_found", extra={"path": str(_cleaned_dir)})
        return {"written": 0, "skipped": 0, "failed": 0, "total": 0}

    paths = list(_cleaned_dir.glob("*.json"))
    stats = {"written": 0, "skipped": 0, "failed": 0, "total": 0}

    logger.info("prepare_embedding_input_start", extra={"files": len(paths)})

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_process_one, p, _output_dir, allowed_show_ids, force): p
            for p in paths
        }
        with tqdm(total=len(paths), desc="Preparing embedding inputs", unit="ep") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result != "filtered":
                    stats["total"] += 1
                if result == "written":
                    stats["written"] += 1
                elif result == "skipped":
                    stats["skipped"] += 1
                elif result == "failed":
                    stats["failed"] += 1
                pbar.update(1)
                pbar.set_postfix(
                    written=stats["written"],
                    skipped=stats["skipped"],
                    failed=stats["failed"],
                )

    logger.info("prepare_embedding_input_complete", extra=stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build embedding_input files from cleaned episode JSON (Layer 2 → Layer 3)"
    )
    parser.add_argument("--show-ids", nargs="+", metavar="SHOW_ID", help="Only process these show IDs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing embedding_input files")
    args = parser.parse_args()

    setup_logging()

    show_ids_filter = set(args.show_ids) if args.show_ids else None
    run(allowed_show_ids=show_ids_filter, force=args.force)


if __name__ == "__main__":
    main()
