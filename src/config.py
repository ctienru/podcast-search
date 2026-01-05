from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ElasticsearchConfig:
    """
    Elasticsearch related configuration.
    """

    url: str = os.getenv("ES_URL", "http://localhost:9200")


@dataclass(frozen=True)
class PathsConfig:
    """
    Repository path configuration.
    All paths are relative to repo root.
    """

    mappings_dir: Path = Path("mappings")
    data_dir: Path = Path("data")

    @property
    def shows_mapping(self) -> Path:
        return self.mappings_dir / "shows.json"

    @property
    def episodes_mapping(self) -> Path:
        return self.mappings_dir / "episodes.json"


@dataclass(frozen=True)
class AppConfig:
    """
    Application-level configuration entrypoint.
    """

    es: ElasticsearchConfig = ElasticsearchConfig()
    paths: PathsConfig = PathsConfig()


# singleton-like access
config = AppConfig()