# podcast-search

Search indexing service that syncs podcast data from crawler output to Elasticsearch for search queries.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ podcast-crawler │────▶│  Local Storage  │────▶│ podcast-search  │
│  (RSS Fetcher)  │     │  (Raw + Shows)  │     │  (Parse+Index)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │  Elasticsearch  │
                                              │ (Search Engine) │
                                              └─────────────────┘
```

### Data Flow

```
podcast-crawler                    podcast-search
     │                                  │
     ├── raw/rss/*.xml ───────────────▶ clean_episodes.py ──────▶ cleaned/episodes/*.json
     │                                  │                                │
     │                                  │                                ▼
     │                                  │                        build_embedding_input.py
     │                                  │                                │
     │                                  │                                ▼
     │                                  │                        embedding_input/episodes/*.json
     │                                  │                                │
     │                                  │                                ▼
     │                                  │                        embed_and_ingest.py ──▶ ES episodes
     │                                  │
     └── normalized/shows/*.json ─────▶ ingest_shows.py ────────────────▶ ES shows
```

## Features

- **Hybrid Search**: BM25 + kNN with RRF (Reciprocal Rank Fusion)
- **Data Cleaning**: RSS parsing, boilerplate removal, multilingual support
- **Embedding Pipeline**: Sentence Transformers for semantic search
- **Search Evaluation**: No-annotation metrics for quality assessment
- **Versioned Indexing**: Zero-downtime schema upgrades via alias switching

## Project Structure

```
podcast-search/
├── src/
│   ├── config/          # Configuration
│   │   └── settings.py  # Environment variables
│   ├── storage/         # Data access layer
│   │   ├── base.py      # Abstract interface
│   │   └── local.py     # Local filesystem
│   ├── es/              # Elasticsearch utilities
│   │   ├── client.py    # ES connection
│   │   └── mapping_loader.py
│   ├── services/        # Service layer
│   │   ├── es_service.py          # ES operations
│   │   └── search_service.py      # BM25/kNN/Hybrid search
│   ├── pipelines/       # ETL Pipelines
│   │   ├── create_indices.py      # Index creation
│   │   ├── ingest_shows.py        # Shows ingestion
│   │   ├── ingest_episodes.py     # Episodes ingestion
│   │   ├── clean_episodes.py      # RSS → Cleaned JSON
│   │   ├── build_embedding_input.py  # Cleaned → Embedding input
│   │   ├── embed_and_ingest.py    # Embedding + ES ingest
│   │   └── evaluate_search.py     # Search evaluation
│   ├── cleaning/        # Text cleaning
│   │   ├── rss_parser.py          # RSS XML parser
│   │   └── text_cleaner.py        # Boilerplate removal
│   ├── embedding/       # Embedding generation
│   │   ├── encoder.py             # Sentence Transformer
│   │   └── input_builder.py       # Title-weighted text
│   ├── evaluation/      # Search evaluation
│   │   ├── extraneous_scorer.py   # Extraneous content detection
│   │   └── metrics.py             # No-annotation metrics
│   ├── api/             # FastAPI endpoints
│   │   ├── main.py
│   │   ├── models.py
│   │   └── routes.py
│   └── utils/
│       └── logging.py   # JSON structured logging
├── mappings/            # ES index definitions
│   ├── shows.json
│   └── episodes.json
├── tests/               # Unit tests
├── docker/              # Local dev environment
│   └── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 2. Start Local Elasticsearch

```bash
docker compose -f docker/docker-compose.yml up -d
```

Services available at:
- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601/app/dev_tools#/console

### 3. Prepare Data

Place crawler's output in the `data/` directory:

```
data/
├── normalized/
│   └── shows/
│       └── show:apple:123.json
└── (raw RSS from crawler at ../podcast-crawler/data/raw/rss/)
```

### 4. Run Pipelines

```bash
# Create indices
python -m src.pipelines.create_indices

# Ingest shows
python -m src.pipelines.ingest_shows

# Clean episodes (raw RSS → cleaned JSON)
python -m src.pipelines.clean_episodes

# Build embedding input (cleaned → embedding format)
python -m src.pipelines.build_embedding_input

# Generate embeddings and ingest to ES
python -m src.pipelines.embed_and_ingest --batch-size 64
```

**Full Reindex (one-liner)**:

```bash
python -m src.pipelines.create_indices && \
python -m src.pipelines.ingest_shows && \
python -m src.pipelines.clean_episodes && \
python -m src.pipelines.build_embedding_input && \
python -m src.pipelines.embed_and_ingest --batch-size 64
```

### 5. Run Evaluation

After indexing, run evaluation to validate search quality:

```bash
# No-Annotation evaluation (Extraneous, Stability, Dominance)
python -m src.pipelines.evaluate_search

# BM25 vs kNN vs Hybrid comparison (Jaccard analysis)
python scripts/compare_search_methods.py --output data/evaluation/method_comparison.json
```

**Full Reindex + Evaluate**:

```bash
python -m src.pipelines.create_indices && \
python -m src.pipelines.ingest_shows && \
python -m src.pipelines.clean_episodes && \
python -m src.pipelines.build_embedding_input && \
python -m src.pipelines.embed_and_ingest --batch-size 64 && \
python -m src.pipelines.evaluate_search && \
python scripts/compare_search_methods.py --output data/evaluation/method_comparison.json
```

### 6. Verify Results

In Kibana Dev Tools:

```
GET shows/_count
GET episodes/_count

GET shows/_search
{
  "query": { "match": { "title": "daily" } }
}
```

## Environment Variables

### Basic Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Local data directory | `data` |
| `ES_HOST` | Elasticsearch URL | `http://localhost:9200` |
| `LOG_LEVEL` | Log level | `INFO` |

### Index Management

| Variable | Description | Default |
|----------|-------------|---------|
| `INDEX_VERSION` | Index version number | `1` |
| `REINDEX` | Reindex on version upgrade | `false` |
| `ALLOW_DELETE_BASE_INDEX` | Allow deleting non-aliased indices | `false` |

### Remote Elasticsearch

| Variable | Description |
|----------|-------------|
| `ES_API_KEY` | ES API Key (for remote auth) |

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/cleaning/test_text_cleaner.py -v
```

## ES Index Structure

### Shows Index

```json
{
  "show_id": "show:apple:123",
  "title": "The Daily",
  "publisher": "The New York Times",
  "description": "This is what the news should sound like...",
  "language": "en",
  "categories": ["News", "News > Daily News"],
  "explicit": false,
  "show_type": "episodic",
  "episode_count": 1500,
  "last_episode_at": "2026-01-18T06:00:00Z",
  "external_ids": { "apple": "123" },
  "external_urls": { "apple": "https://podcasts.apple.com/us/podcast/..." },
  "image_url": "https://is1-ssl.mzstatic.com/image/...",
  "created_at": "2026-01-14T12:00:00Z",
  "updated_at": "2026-01-18T06:00:00Z"
}
```

### Episodes Index

```json
{
  "episode_id": "episode:apple:123:ep1",
  "title": "Episode Title",
  "description": "Cleaned episode description (boilerplate removed)...",
  "embedding": [0.1, 0.2, ...],
  "published_at": "2026-01-18T06:00:00Z",
  "duration_sec": 1800,
  "language": "en",
  "show": {
    "show_id": "show:apple:123",
    "title": "The Daily",
    "publisher": "The New York Times",
    "image_url": "https://is1-ssl.mzstatic.com/image/...",
    "external_urls": {
      "apple_podcasts": "https://podcasts.apple.com/us/podcast/..."
    }
  },
  "audio": {
    "url": "https://..."
  },
  "created_at": "2026-01-18T06:00:00Z",
  "updated_at": "2026-01-18T06:00:00Z"
}
```

---

## Search Features

### Hybrid Search (BM25 + kNN + RRF)

Supports three search modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `bm25` | Pure text matching | Keyword search |
| `knn` | Pure semantic search | Concept search |
| `hybrid` | BM25 + kNN + RRF fusion | **Recommended** |

```python
from src.services.search_service import SearchService, SearchMode

service = SearchService()

# Hybrid search (recommended)
results = service.search_hybrid("podcast about AI", size=10)

# BM25 only
results = service.search_bm25("tech news", size=10)

# kNN only
results = service.search_knn("machine learning tutorials", size=10)
```

**RRF (Reciprocal Rank Fusion)**:
- Combines BM25 and kNN rankings without score normalization
- Manual implementation (ES RRF requires paid license)
- Formula: `score = sum(1 / (rank_constant + rank_i))`

---

## Search Evaluation

No-annotation evaluation system to validate search quality without manual labeling.

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Same-Podcast Dominance | % of top-K from same show | < 50% |
| Extraneous Intrusion | % of top-K with ad content | < 10% |
| Query Perturbation Stability | Robustness to query variations | > 70% |

### Running Evaluation

```bash
# Full evaluation (30 queries)
python -m src.pipelines.evaluate_search

# Custom queries
python -m src.pipelines.evaluate_search --queries data/test_queries.txt

# Output to specific path
python -m src.pipelines.evaluate_search --output data/evaluation/report.json
```

### Sample Output

```
============================================================
Search Quality Evaluation Report
============================================================
Metrics Summary:
  Same-Podcast Dominance:   23.00%  PASS
  Extraneous Intrusion:      0.00%  PASS
  Query Perturbation Stability: 81.00%  PASS

Overall: PASS
============================================================
```

### Components

- **ExtraneousScorer**: Detects sponsor/CTA/promo content in paragraphs
- **NoAnnotationEvaluator**: Computes 4 metrics without ground truth labels
- **EvaluationPipeline**: Runs evaluation and generates reports

---

## Data Cleaning Pipeline

Three-layer architecture for podcast text cleaning:

```
Layer 1: Raw RSS XML  → From podcast-crawler (../podcast-crawler/data/raw/rss/)
Layer 2: Cleaned JSON → Boilerplate removed, normalized (data/cleaned/episodes/)
Layer 3: Embedding    → Title-weighted, truncated for embedding (data/embedding_input/)
```

### Data Mapping from Crawler

The search service handles data mapping from crawler output:

**Shows (`ingest_shows.py`)**:
- Maps `provider` field to external ID/URL keys
- Crawler uses `{"provider": "apple", "external_urls": {"apple_podcasts": "..."}}`
- Search normalizes to `{"external_ids": {"apple": "..."}, "external_urls": {"apple": "..."}}`
- Extracts `image.url` → `image_url`

**Episodes (`embed_and_ingest.py`)**:
- Embeds show metadata into episode documents
- Includes `show.image_url` and `show.external_urls` for frontend display
- Sources show data from `data/normalized/shows/*.json`

### Cleaning Features

- **HTML → Plain Text**: BeautifulSoup parsing, entity decoding
- **Boilerplate Removal**: Sponsors, CTAs, hosting info, production credits
- **Frequency Filtering**: Removes paragraphs appearing in >25% of episodes (fixed intro/outro)
- **Multilingual Support**: Chinese and English pattern matching

### Running Data Pipelines

```bash
# Step 1: Clean episodes (Layer 1 → Layer 2)
python -m src.pipelines.clean_episodes

# Step 2: Build embedding input (Layer 2 → Layer 3)
python -m src.pipelines.build_embedding_input

# Step 3: Generate embeddings and ingest to ES
python -m src.pipelines.embed_and_ingest --batch-size 64
```

---

## Common Kibana Queries

```bash
# Check index status
GET _cat/indices?v

# Check aliases
GET _aliases

# Search shows
GET shows/_search
{
  "query": { "match": { "title": "podcast" } }
}

# Episodes autocomplete
GET episodes/_search
{
  "query": {
    "match": {
      "title.autocomplete": "the d"
    }
  }
}

# Filter episodes by show
GET episodes/_search
{
  "query": {
    "term": { "show.show_id": "show:apple:123" }
  }
}

# Hybrid search with kNN
GET episodes/_search
{
  "query": {
    "bool": {
      "should": [
        { "multi_match": { "query": "AI", "fields": ["title^3", "description"] } }
      ]
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [0.1, 0.2, ...],
    "k": 10,
    "num_candidates": 100
  }
}
```

---

## Related Projects

- **podcast-crawler**: RSS fetching service (raw XML + show metadata)
- **podcast-api**: Search API service
- **podcast-frontend**: Frontend interface
