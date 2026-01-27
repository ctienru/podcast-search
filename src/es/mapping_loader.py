import json
from pathlib import Path
from typing import Any, Dict


class MappingLoader:
    """
    Load index mapping JSON files from repo-local mappings/ directory.
    This is NOT a storage backend (mappings are static repo assets).
    """

    def __init__(self, mappings_dir: Path):
        self.mappings_dir = mappings_dir

    def load(self, index_name: str) -> Dict[str, Any]:
        path = self.mappings_dir / f"{index_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"mapping_not_found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))