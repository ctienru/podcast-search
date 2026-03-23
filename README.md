# podcast-search

Search indexing service that syncs podcast data from the crawler's SQLite database and RSS feeds into Elasticsearch for multi-language episode search.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────────┐
│ podcast-crawler │────▶│  SQLite (crawler.db)  │
│  (RSS Fetcher)  │     │  + raw/rss/*.xml       │
└─────────────────┘     └──────────┬───────────┘
                                   │
                          ┌────────▼────────┐
                          │ podcast-search  │
                          │ (Parse + Index) │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             episodes-zh-tw  episodes-zh-cn  episodes-en
                    └──────────────┴──────────────┘
                               Elasticsearch
```

### Data Flow

```
podcast-crawler
     │
     ├── SQLite (shows: language, target_index)
     │        │
     │        └─────▶ ingest_shows.py ──────────────▶ ES shows
     │
     └── raw/rss/*.xml
              │
              └─────▶ clean_episodes.py ─▶ data/cleaned/episodes/
                             │
                             └─▶ embed_and_ingest.py ─▶ ES episodes-{lang}
                                  (embed + route by language)
```

## Project Structure

```
podcast-search/
├── src/
│   ├── config/
│   │   └── settings.py          # Environment variables
│   ├── types.py                 # Shared type definitions (Language, Show, Episode, ...)
│   ├── storage/                 # Data access layer (Repository pattern)
│   │   ├── base.py              # StorageBase ABC
│   │   ├── sqlite.py            # SQLiteStorage (v2 primary source)
│   │   ├── local.py             # LocalStorage (dev fallback)
│   │   └── factory.py           # StorageFactory
│   ├── search/
│   │   └── routing.py           # IndexRoutingStrategy (Strategy pattern)
│   ├── es/                      # Elasticsearch utilities
│   │   ├── client.py
│   │   ├── index_creator.py
│   │   └── mapping_loader.py
│   ├── services/
│   │   ├── es_service.py        # ES bulk operations
│   │   └── search_service.py    # BM25/kNN/Hybrid search (evaluation use)
│   ├── pipelines/               # ETL pipelines
│   │   ├── create_indices.py    # Create 3 language indices + aliases
│   │   ├── ingest_shows.py      # Shows: SQLite → ES shows index
│   │   ├── clean_episodes.py    # RSS XML → cleaned JSON
│   │   ├── embed_and_ingest.py  # Cleaned JSON → embedding → ES episodes
│   │   └── evaluate_search.py   # Search quality evaluation pipeline
│   ├── embedding/
│   │   └── backend.py           # EmbeddingBackend ABC, LocalEmbeddingBackend, APIEmbeddingBackend
│   ├── cleaning/
│   │   ├── rss_parser.py        # RSS XML parser
│   │   └── text_cleaner.py      # Boilerplate removal
│   ├── evaluation/              # Search quality metrics
│   │   ├── query_logger.py      # QueryLogger (Middleware pattern)
│   │   ├── click_tracker.py     # ClickTracker (Middleware pattern)
│   │   ├── metrics.py           # No-annotation metrics
│   │   ├── ranking_metrics.py   # NDCG, MRR
│   │   ├── cross_encoder_judge.py
│   │   └── extraneous_scorer.py
│   ├── api/                     # FastAPI endpoints
│   │   ├── main.py
│   │   ├── models.py
│   │   └── routes.py            # GET /health, POST /embed
│   └── utils/
│       ├── logging.py           # JSON structured logging
│       └── parsers.py           # RSS metadata parsing utilities
├── mappings/
│   ├── episodes-zh-tw.json      # Traditional Chinese index (IK analyzer)
│   ├── episodes-zh-cn.json      # Simplified Chinese index (IK analyzer)
│   ├── episodes-en.json         # English index (standard analyzer)
│   └── shows.json               # Shows index
├── scripts/
│   ├── migrate_reindex.py       # Phase 2: data migration to 3-index layout
│   ├── compute_online_metrics.py
│   ├── check_regression_gate.py
│   ├── compare_search_methods.py
│   ├── evaluate_ndcg_mrr.py
│   └── build_annotation_pool.py
├── tests/
│   ├── unit/                    # Unit tests (PR gate)
│   ├── integration/             # Integration tests (require real ES)
│   ├── cleaning/
│   ├── evaluation/
│   ├── pipelines/
│   ├── services/
│   └── storage/
├── docker/
│   └── docker-compose.yml       # Local Elasticsearch + Kibana
└── requirements.txt
```

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 2. Start Local Elasticsearch

```bash
docker compose -f docker/docker-compose.yml up -d
```

Services:
- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601/app/dev_tools#/console

### 3. Data Sources

```
data/
└── crawler.db                        # SQLite from podcast-crawler v2
../podcast-crawler/data/raw/rss/      # RSS XML files (shared with crawler)
```

### 4. Full Re-index (fresh ES)

Use this when starting from a clean ES instance or after a mapping change.

```bash
# Step 1 — Create 3 language indices + aliases (one-time)
python -m src.pipelines.create_indices

# Step 2 — Ingest shows from SQLite
python -m src.pipelines.ingest_shows

# Step 3 — Clean episodes: RSS XML → data/cleaned/episodes/
python -m src.pipelines.clean_episodes

# Step 4 — Embed and ingest all episodes (full mode, re-embeds everything)
SYNC_MODE=full python -m src.pipelines.embed_and_ingest
```

### 5. Verify

```
GET _aliases
GET episodes-zh-tw/_count
GET episodes-zh-cn/_count
GET episodes-en/_count
GET shows/_count
```

---

## Run Scenarios

### Daily incremental sync (default)

Only processes shows updated since the last run. Uses cursor in `data/ingest_cursor.json`.

```bash
python -m src.pipelines.ingest_shows
python -m src.pipelines.clean_episodes
python -m src.pipelines.embed_and_ingest
# SYNC_MODE defaults to "incremental"
```

### Backfill a newly added non-analyzer field

Use after adding a new field to the mapping that doesn't require re-tokenization (e.g. `image_url`, `episode_count`). Re-ingests all docs but **skips re-embedding** — existing vectors are preserved via ES update.

```bash
SYNC_MODE=backfill python -m src.pipelines.embed_and_ingest
```

### Force full re-ingest + re-embed

Use after a mapping change that affects analyzers, or after switching embedding models.

```bash
SYNC_MODE=full python -m src.pipelines.embed_and_ingest
```

### Re-ingest a single show

Use to fix a specific show without touching anything else.

```bash
python -m src.pipelines.embed_and_ingest --show-id show:apple:12345678
```

### Re-index with new embedding model (alias switch pattern)

Use after upgrading the embedding model (e.g. Phase 3-A: bge-zh upgrade).

```bash
# 1. Bump INDEX_VERSION to create a new backing index
INDEX_VERSION=2 python -m src.pipelines.create_indices

# 2. Full re-embed into the new index
INDEX_VERSION=2 SYNC_MODE=full python -m src.pipelines.embed_and_ingest

# 3. Verify counts match, run regression gate
python scripts/check_regression_gate.py

# 4. Switch alias (run in Kibana)
# POST /_aliases
# { "actions": [
#     { "remove": { "index": "podcast-episodes-zh-tw-v1", "alias": "episodes-zh-tw" } },
#     { "add":    { "index": "podcast-episodes-zh-tw-v2", "alias": "episodes-zh-tw" } }
# ] }
```

## API Endpoints

The service exposes two endpoints used by `podcast-backend`:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /embed` | Encode a query string to an embedding vector |

```bash
# Embed a query
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"query": "人工智慧", "language": "zh-tw"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SQLITE_PATH` | `data/crawler.db` | SQLite path (v2 data source) |
| `DATA_DIR` | `data` | Local data directory |
| `ES_HOST` | `http://localhost:9200` | Elasticsearch URL |
| `ES_API_KEY` | — | ES API key (remote auth) |
| `INDEX_VERSION` | `1` | Index version number |
| `LOG_LEVEL` | `INFO` | Log level |
| `SYNC_MODE` | `incremental` | `full` / `incremental` / `backfill` / `single` |
| `QUERY_LOG_PATH` | `logs/query_log.jsonl` | Query behavioral log |
| `CLICK_LOG_PATH` | `logs/click_log.jsonl` | Click behavioral log |
| `INGEST_CURSOR_PATH` | `data/ingest_cursor.json` | Incremental ingest cursor |
| `EMBEDDING_API_URL` | — | External embedding API URL (Phase 3-B) |
| `EMBEDDING_API_KEY` | — | External embedding API key (Phase 3-B) |

## ES Index Structure

### Episodes Indices (3 language-split indices)

```
alias              → backing index
episodes-zh-tw     → podcast-episodes-zh-tw-v{N}    (Traditional Chinese, IK analyzer)
episodes-zh-cn     → podcast-episodes-zh-cn-v{N}    (Simplified Chinese, IK analyzer)
episodes-en        → podcast-episodes-en-v{N}        (English, standard analyzer)
```

Episode document:
```json
{
  "episode_id": "...",
  "title": "Episode Title",
  "description": "Cleaned description...",
  "embedding": [0.1, 0.2, ...],
  "published_at": "2026-01-18T06:00:00Z",
  "duration_sec": 1800,
  "language": "zh-tw",
  "show": {
    "show_id": "...",
    "title": "Show Title",
    "publisher": "Author Name",
    "image_url": "https://example.com/cover.jpg",
    "external_urls": { "apple_podcasts": "https://podcasts.apple.com/..." }
  },
  "audio": { "url": "https://..." }
}
```

### Shows Index

```json
{
  "show_id": "...",
  "title": "Show Title",
  "publisher": "Author Name",
  "description": "Show description...",
  "language": "zh-tw",
  "image_url": "https://example.com/cover.jpg",
  "external_ids": { "apple_podcasts": "12345678" },
  "external_urls": { "apple_podcasts": "https://podcasts.apple.com/..." },
  "categories": ["Technology", "Technology > AI"],
  "episode_count": 150,
  "last_episode_at": "2026-03-20T00:00:00Z",
  "updated_at": "2026-03-20T00:00:00Z"
}
```

## Embedding Architecture

| Use case | Backend | Model |
|----------|---------|-------|
| Index-time (zh-tw, zh-cn) | `LocalEmbeddingBackend` | `BAAI/bge-base-zh-v1.5` (768 dim) |
| Index-time (en) | `LocalEmbeddingBackend` | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| Query-time | `POST /embed` → `LocalEmbeddingBackend` | Same as index-time per language |

**Fallback:** If `EMBEDDING_API_URL` is set, `APIEmbeddingBackend` is used at query-time instead (Phase 3-B, triggered if LRU cache hit rate < 60%).

## Search Evaluation

```bash
# No-annotation evaluation (Extraneous, Stability, Dominance)
python -m src.pipelines.evaluate_search

# NDCG/MRR comparison
python scripts/evaluate_ndcg_mrr.py

# BM25 vs kNN vs Hybrid comparison
python scripts/compare_search_methods.py

# Check regression gate (exits non-zero if NDCG drops > 5%)
python scripts/check_regression_gate.py
```

### Offline Regression Gate

The PR gate (`.github/workflows/pr-gate.yml`) runs unit tests and the NDCG regression gate on every PR. Merge is blocked if:
- Any unit test fails
- Overall NDCG@10 drops below baseline (zh ≥ 0.897, en ≥ 0.853)

### Online Behavioral Metrics

Query logs (`logs/query_log.jsonl`) and click logs (`logs/click_log.jsonl`) are written at runtime by `QueryLogger` and `ClickTracker`. Compute metrics:

```bash
python scripts/compute_online_metrics.py \
  --query-log logs/query_log.jsonl \
  --click-log logs/click_log.jsonl
```

## Testing

```bash
# Unit tests (fast, no ES required)
pytest tests/unit/ -v

# Integration tests (require running ES)
pytest tests/integration/ -v -m integration

# All non-integration tests with coverage
pytest --ignore=tests/integration -q --cov=src
```

## Related Projects

- **podcast-crawler**: RSS fetching service (produces `crawler.db` + raw RSS XML)
- **podcast-backend**: Search API consumed by frontend (routes queries to this service's `/embed`)
- **podcast-frontend**: Frontend interface
