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
     │        └─────▶ ingest_shows.py ──────────────────────────────▶ ES shows
     │
     └── raw/rss/*.xml
              │
              └─────▶ clean_episodes.py ─▶ data/cleaned/episodes/
                             │
                             ├─▶ embed_and_ingest.py ─────────────▶ ES episodes-{lang}
                             │    (embed inline + route by language)
                             │
                             └─▶ embed_episodes.py ─▶ data/embeddings/
                                  (Step 1: pre-compute vectors, no ES)
                                        │
                                        └─▶ embed_and_ingest --from-cache ─▶ ES episodes-{lang}
                                             (Step 2: read cache, write ES)
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
│   │   ├── factory.py           # StorageFactory
│   │   ├── sync_state.py        # SyncStateRepository — writes search_sync_state after ES ingest
│   │   └── episode_status.py    # EpisodeStatusRepository — writes embedding_status after embed
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
│   │   ├── embed_episodes.py    # Cleaned JSON → embedding vectors → data/embeddings/ (no ES)
│   │   ├── embed_and_ingest.py  # Cleaned JSON → embedding → ES episodes (or --from-cache)
│   │   ├── exceptions.py        # Pipeline exceptions (CacheMissError)
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
├── evaluation/
│   └── ndcg_mrr_baseline.json       # Committed NDCG baseline for PR gate
├── scripts/
│   ├── migrate_reindex.py               # Phase 2: data migration to 3-index layout
│   ├── evaluate_ndcg_mrr.py             # Offline Quality: NDCG@10 + MRR per language/method
│   ├── check_regression_gate.py         # Offline Quality: pass/fail check against NDCG thresholds
│   ├── evaluate_language_detection.py   # Correctness: language routing accuracy (RSS tag → index)
│   ├── index_health_report.py           # Correctness: embedding coverage, zero-result rate
│   ├── benchmark_latency.py             # Offline Quality: P50/P95/P99 latency per language/mode
│   ├── run_evaluation_suite.py          # Run Correctness + Offline Quality checks in sequence
│   ├── generate_weekly_report.py        # Weekly report (Offline Quality + Online Behavior)
│   ├── compute_online_metrics.py        # Online Behavior: SSR, same-language click rate, reformulation rate
│   ├── compare_search_methods.py        # BM25 vs kNN vs Hybrid overlap/rank comparison
│   ├── build_annotation_pool.py
│   └── annotate_with_cross_encoder.py
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
├── crawler.db                        # SQLite from podcast-crawler v2
├── cleaned/episodes/                 # Cleaned episode JSON (output of clean_episodes)
└── embeddings/                       # Pre-computed embedding vectors (output of embed_episodes)
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

# Step 4a — Embed and ingest inline (single step, requires local GPU/model)
SYNC_MODE=full python -m src.pipelines.embed_and_ingest

# Step 4b — Two-step: pre-compute vectors locally, then write to remote ES
python -m src.pipelines.embed_episodes          # slow, local; writes data/embeddings/
ES_HOST=<remote> SYNC_MODE=full python -m src.pipelines.embed_and_ingest --from-cache
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

`embed_and_ingest.py` uses `data/ingest_cursor.json` for incremental sync. `ingest_shows.py` is still full-sync (no cursor yet).

```bash
python -m src.pipelines.ingest_shows
python -m src.pipelines.clean_episodes
python -m src.pipelines.embed_and_ingest
# SYNC_MODE defaults to "incremental" (for embed_and_ingest)
```

### Two-step workflow (decouple embedding from ES)

Use when the embedding model runs locally but ES is remote. Step 1 computes vectors offline and writes `embedding_status='done'` back to SQLite; Step 2 only touches the network.

```bash
# Step 1 (local, slow): compute vectors, cache to data/embeddings/, write embedding_status='done'
python -m src.pipelines.embed_episodes

# Step 2 (fast, remote ES): read cache, write to ES, write search_sync_state(environment='local')
ES_ENV=local ES_HOST=<remote> python -m src.pipelines.embed_and_ingest --from-cache

# Step 3 (production ES): same cache, different ES, writes separate search_sync_state row
ES_ENV=production ES_HOST=<prod> ES_API_KEY=<key> python -m src.pipelines.embed_and_ingest --from-cache

# Strict mode: fail immediately if any episode is missing from the cache
ES_ENV=local ES_HOST=<remote> python -m src.pipelines.embed_and_ingest --from-cache --strict-cache
```

After Step 2 + 3, `search_sync_state` holds two independent rows per episode — one for each environment — so both can be tracked independently.

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
| `POST /embed` | Encode a query string to an embedding vector (internal format) |
| `POST /v1/embeddings` | OpenAI-compatible embeddings endpoint (used by `podcast-backend` EmbeddingProvider) |

```bash
# Embed a query (internal format)
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["artificial intelligence"], "language": "zh-tw"}'

# Embed via OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["artificial intelligence"], "model": "BAAI/bge-base-zh-v1.5"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SQLITE_PATH` | `data/crawler.db` | SQLite path (v2 data source) |
| `DATA_DIR` | `data` | Local data directory |
| `ES_HOST` | `http://localhost:9200` | Elasticsearch URL |
| `ES_ENV` | `default` | ES environment label written to `search_sync_state.environment` — use `local` or `production` to track multi-env syncs independently |
| `ES_API_KEY` | — | ES API key (remote auth) |
| `INDEX_VERSION` | `1` | Index version number |
| `LOG_LEVEL` | `INFO` | Log level |
| `SYNC_MODE` | `incremental` | `full` / `incremental` / `backfill` / `single` |
| `QUERY_LOG_PATH` | `logs/query_log.jsonl` | Query behavioral log |
| `CLICK_LOG_PATH` | `logs/click_log.jsonl` | Click behavioral log |
| `INGEST_CURSOR_PATH` | `data/ingest_cursor.json` | Incremental ingest cursor |
| `ENABLE_LANGUAGE_SPLIT` | `true` | Use 3-index language-split layout (v2 default) |
| `EMBEDDING_CACHE_DIR` | `data/embeddings` | Vector cache dir (written by `embed_episodes`, read by `embed_and_ingest --from-cache`) |
| `EMBEDDING_TEXT_VERSION` | `text-v1` | Embedding text assembly version; bump when the text format changes |
| `EMBEDDING_API_URL` | — | External embedding API URL (Phase 3-B) |
| `EMBEDDING_API_KEY` | — | External embedding API key (Phase 3-B) |

## ES Index Structure

### Index Naming Convention

All podcast indices follow `podcast-{type}-{qualifier}_v{N}`:

```
alias              → backing index
shows              → podcast-shows_v{N}
episodes-zh-tw     → podcast-episodes-zh-tw_v{N}    (Traditional Chinese, IK analyzer)
episodes-zh-cn     → podcast-episodes-zh-cn_v{N}    (Simplified Chinese, IK analyzer)
episodes-en        → podcast-episodes-en_v{N}        (English, standard analyzer)
```

Backend always uses alias names — never the backing index name directly.

### Episodes Indices (3 language-split indices)

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
| Query-time | `APIEmbeddingBackend` (`/embed` endpoint) | Must match index-time model |

Index-time uses `LocalEmbeddingBackend` (language-to-model mapping defined in `MODEL_MAP`). Query-time goes through the `/embed` endpoint backed by `APIEmbeddingBackend`, which is lazy-loaded on first request and shared for the process lifetime. Index-time and query-time must use the same model — mixing backends produces incomparable vector spaces and silently degrades kNN quality.

## Search Evaluation

Evaluation is structured as three check groups. See `podcast-doc/v2/2026-03-25-search-evaluation-v2.md` for full framework details.

### Correctness (run before deploy)

Data quality checks — are the indices healthy and is routing correct?

```bash
# Index health: embedding coverage ≥ 99%, zero-result rate < 5%
python scripts/index_health_report.py --output data/evaluation/index_health.json

# Language routing accuracy: ≥ 95% precision, ≥ 90% recall
# Requires labeled sample at data/evaluation/language_detection_sample.json
python scripts/evaluate_language_detection.py
```

**Current status**: Index health PASS. Language routing FAIL — 3 shows have wrong RSS `<language>` tags routed to `en` index; fix is in the crawler/ingest layer.

### Offline Quality (run before deploy and on PR)

Search quality checks — does ranking meet regression thresholds?

```bash
# Run the full suite (Correctness + Offline Quality, ~4 min, requires local ES)
python scripts/run_evaluation_suite.py
# Output: data/evaluation/reports/{date}/suite_report.json

# Run only the NDCG evaluation
python scripts/evaluate_ndcg_mrr.py \
    --output data/evaluation/reports/$(date +%Y-%m-%d)/ndcg_mrr_report.json

# Check regression against thresholds: zh-tw ≥ 0.897 | zh-cn ≥ 0.897 (provisional) | en ≥ 0.853
python scripts/check_regression_gate.py --report evaluation/ndcg_mrr_baseline.json

# Latency baseline (full, 20 runs — use to rebuild baseline)
python scripts/benchmark_latency.py --runs 20 --output data/benchmark/latency_baseline.json

# Latency quick check (5 runs — used by suite)
python scripts/benchmark_latency.py --quick
```

**PR Gate**: `pr-gate.yml` runs `check_regression_gate.py` against `evaluation/ndcg_mrr_baseline.json` in warning mode (`continue-on-error: true`). To update the baseline after a suite run:

```bash
cp data/evaluation/reports/$(date +%Y-%m-%d)/ndcg_mrr_report.json evaluation/ndcg_mrr_baseline.json
# then commit evaluation/ndcg_mrr_baseline.json
```

### Online Behavior (weekly, post-launch)

Behavioral metrics from real user traffic.

```bash
python scripts/compute_online_metrics.py \
  --query-log logs/query_log.jsonl \
  --click-log logs/click_log.jsonl
```

Thresholds: Search Success Rate ≥ 60%, Same-Language Click Rate ≥ 80%, Reformulation Rate ≤ 20%.

### Weekly Report

```bash
python scripts/generate_weekly_report.py \
    --date $(date +%Y-%m-%d) \
    --prev-date YYYY-MM-DD \
    --query-log logs/query_log.jsonl \
    --click-log logs/click_log.jsonl \
    --output reports/weekly_$(date +%Y-%m-%d).md
```

Produces a Markdown report combining Offline Quality method comparison, per-query NDCG delta vs previous run, and Online Behavior metrics.

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
