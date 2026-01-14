# Elasticsearch Mapping – Episodes Index

This document defines the Elasticsearch index mapping
for **podcast episodes**.

The mapping is designed for **intent-driven episode search**:
high precision, contextual results, and predictable performance.

This file is written in plain Markdown for easy copy, review,
and long-term maintenance.

---

## Index Purpose

The `episodes` index supports **episode-level search**.

Typical user intent:
- Search for a specific topic
- Find a particular episode
- Browse discussions within podcasts

Compared to show discovery:
- Precision is more important than recall
- Contextual snippets (highlight) are required
- Pagination and sorting are critical

---

## Identifier Strategy

Episodes use a **dual-layer identifier model**.

### Internal Identifier

| Field | Type | Description |
|------|------|-------------|
| episode_id | keyword | System-generated internal episode ID |

Internal identifiers are:
- Used in API responses
- Used as Elasticsearch `_id`
- Used in URLs and cache keys
- Stable and independent of external sources

---

### External Identifiers (Multi-source)

External identifiers are stored in a **controlled object map**.

| Field | Type | Indexed | Description |
|------|------|---------|-------------|
| external_ids | object | Yes | Map of `{ source -> external_id }` |

Example:
```
external_ids:
  apple_podcasts: “987654321”
  spotify: “spotify:episode:abcd”
```

Rules:
- Keys represent data sources (e.g. `apple_podcasts`, `spotify`)
- Values must be strings
- One ID per source only
- Nested objects or arrays are not allowed

Elasticsearch enforces type safety using `dynamic_templates`.
Allowed source names are validated at the ingestion layer.

---

## Mapping Summary

### Core Episode Fields

| Field | Type | Indexed | Required | Description |
|------|------|---------|----------|-------------|
| episode_id | keyword | Yes | Yes | Stable internal episode identifier |
| title | text | Yes | Yes | Episode title (primary search field) |
| description | text | Yes | No | Episode description |
| language | keyword | Yes | Yes | Episode language (normalized) |
| published_at | date | Yes | Yes | Publish datetime |
| duration_sec | integer | Yes | No | Episode duration (seconds) |
| image_url | keyword | No | No | Episode image (UI only) |

---

### Embedded Podcast Snapshot

Episodes store a **denormalized snapshot** of podcast (show) metadata.

| Field | Type | Indexed | Description |
|------|------|---------|-------------|
| podcast.podcast_id | keyword | Yes | Internal podcast identifier |
| podcast.external_ids | object | Yes | External IDs map (same rules as episode) |
| podcast.title | text | Yes | Podcast title (search + display) |
| podcast.publisher | text | Yes | Podcast publisher |
| podcast.image_url | keyword | No | Podcast image (UI only) |

This snapshot exists to:
- Avoid join queries
- Enable single-pass search
- Improve relevance scoring
- Simplify API response mapping

---

### Search Optimization Fields

| Field | Type | Purpose |
|------|------|---------|
| title.keyword | keyword | Exact match / sorting (optional) |
| title.autocomplete | text | Prefix / type-ahead search (optional) |

Autocomplete fields are optional and should only be enabled
if required by the UI.

---

## Design Principles

### Denormalization Over Joins

This index intentionally duplicates a small subset of podcast data.

Reasons:
- Elasticsearch joins (parent-child) are expensive and complex
- Runtime joins increase latency and failure modes
- Search relevance and sorting work best within a single document

This is a **deliberate trade-off**.

---

### Snapshot Consistency Model

The embedded podcast data is a **snapshot**, not a source of truth.

- It may become temporarily stale
- It is acceptable for search use cases
- Strong consistency is not required

Canonical podcast data lives in:
- The primary database
- The `shows` Elasticsearch index

---

## Podcast Metadata Change Strategy

Podcast metadata (e.g. show name) may change over time.

### Fields That Matter for Episodes

Only the following podcast fields are embedded and maintained:

| Field | Reason |
|------|-------|
| podcast.title | Affects search relevance and display |
| podcast.publisher | Affects search relevance and display |
| podcast.image_url | Affects UI rendering |

Other podcast fields are intentionally excluded.

---

### Update Strategy

The system follows an **eventual consistency** model:

- New episodes always use the latest podcast metadata
- Existing episodes may keep older snapshots
- Periodic reindexing may be performed when important
  podcast fields change

This approach avoids expensive real-time fan-out updates
while keeping search results acceptable.

---

## Reindexing Options

Possible strategies (choose based on scale):

| Strategy | Description | Cost |
|--------|-------------|------|
| No reindex | Only new episodes get updated data | Lowest |
| Scheduled batch | Periodic update by `podcast_id` | Medium |
| Full reindex | Rebuild entire index | Highest |

Batch reindexing is recommended if podcast renames
become frequent or highly visible.

---

## Relationship to Search Query

This mapping is designed to work with:

- Multi-field relevance queries
- Highlighting on `title` and `description`
- Language-based filtering
- Sorting by relevance or publish date

Mapping and query design must evolve together.

---

## Design Boundaries

This index intentionally does **not**:
- Store full podcast metadata
- Guarantee real-time consistency with show updates
- Support relational joins

These constraints are required to keep search fast and reliable.

---

## Summary

The `episodes` index mapping is:

- Optimized for precision search
- Denormalized by design
- Highlight-friendly
- Multi-source ready
- Structurally constrained
- Cache- and performance-aware

This design prioritizes **search quality and system stability**
over strict data normalization.