# Elasticsearch Mapping – Shows Index

This document defines the Elasticsearch index mapping for **podcast shows**.
It is designed for **discovery search** (fast UI rendering, stable fields, and future ranking).

This file is intentionally written in plain Markdown (tables only)
for easy copy/paste, review, and long-term maintenance.

---

## Index Purpose

The `shows` index is optimized for **search and discovery**, not as a full data warehouse.

It stores only fields that are:
- required for search relevance
- required for ranking
- required for UI rendering
- useful for future evolution (without breaking changes)

---

## Identifier Strategy

Shows use a **dual-layer identifier model**.

### Internal Identifier (Primary)

| Field | Type | Indexed | Required | Description |
|------|------|---------|----------|-------------|
| podcast_id | keyword | Yes | Yes | System-generated, stable internal show ID |

Internal identifiers are:
- used in API responses
- used as Elasticsearch `_id`
- used in URLs and cache keys
- independent of external content sources

---

### External Identifiers (Multi-source)

External identifiers are stored in a **controlled object map**.

| Field | Type | Indexed | Required | Description |
|------|------|---------|----------|-------------|
| external_ids | object | Yes | No | Map of `{ source -> external_id }` |

Example:
```
external_ids:
    apple_podcasts: “123456789”
    spotify: “spotify:show:abcd”
    rss: “https://example.com/feed.xml”
```
Rules:
- Keys represent data sources (e.g. `apple_podcasts`, `spotify`)
- Values must be **strings**
- Only one ID per source is allowed
- Nested objects or arrays are not allowed

Elasticsearch enforces structural and type constraints
via `dynamic_templates`.
Allowed source names are validated at the ingestion layer.

External identifiers are used for ingestion and synchronization,
not as primary identifiers in APIs.

---

## Mapping Summary

### Core Fields

| Field | Type | Indexed | Required | Description |
|------|------|---------|----------|-------------|
| podcast_id | keyword | Yes | Yes | Stable internal identifier for the show |
| title | text | Yes | Yes | Main searchable title |
| description | text | Yes | No | Searchable summary |
| publisher | text | Yes | No | Creator / publisher name |
| language | keyword | Yes | Yes | Primary language (normalized, BCP 47-like) |
| episode_count | integer | Yes | No | Number of episodes (ranking signal) |
| popularity_score | float | Yes | No | Composite ranking signal (reserved) |
| image_url | keyword | No | Yes | UI image URL (not indexed) |
| created_at | date | Yes | No | Ingestion timestamp |
| updated_at | date | Yes | No | Last updated timestamp |

---

## Field Definitions

### Identifier Fields

| Field | Type | Notes |
|------|------|------|
| podcast_id | keyword | Internal, stable ID used across the system |
| external_ids | object | Multi-source external ID map (see rules above) |

---

### Searchable Text Fields

| Field | Type | Notes |
|------|------|------|
| title | text | Primary relevance signal (highest boost in query) |
| description | text | Secondary relevance signal |
| publisher | text | Useful when users remember the creator or brand |

Recommended:
`title` and `publisher` may include a `.keyword` subfield
if exact match or sorting is required in the future.

---

### Filter Fields

| Field | Type | Notes |
|------|------|------|
| language | keyword | Must be keyword for `terms` filter |

Language values should be normalized
(e.g. `en`, `zh`, `zh-TW`).

---

### Ranking / Activity Fields

| Field | Type | Notes |
|------|------|------|
| episode_count | integer | Activity / scale signal |
| popularity_score | float | Reserved for ranking optimization |

These fields enable ranking improvements
without changing the API contract.

---

### UI-only Fields

| Field | Type | Notes |
|------|------|------|
| image_url | keyword (index=false) | Required for carousel rendering; not used for search |

We intentionally do not index `image_url`
to reduce index size and search overhead.

---

### Metadata Fields

| Field | Type | Notes |
|------|------|------|
| created_at | date | Optional, for tracing ingestion |
| updated_at | date | Useful for freshness-based ranking later |

---

## Popularity Score Design

### Goal

`popularity_score` represents how suitable a show is for discovery.
It should improve ordering **after** text relevance (`_score`).

We do NOT depend on user behavior in the initial design.
Only ingestion-derived signals are used.

---

### Selected Approach (Planned)

The first practical version will use **update activity**
(freshness and frequency).

#### Signals

| Signal | Description | How to get it |
|------|-------------|---------------|
| last_episode_date | Publish date of latest episode | From episode ingestion |
| days_since_last_episode | Days since last episode | Derived |
| episode_count | Total episodes | From ingestion |

Optional (later):
- average publish interval
- rolling 30 / 90 day episode count

---

### Suggested Formula (Design Draft)

| Term | Definition |
|------|------------|
| freshness_factor | exp( - days_since_last_episode / 30 ) |
| volume_factor | log10(1 + episode_count) |
| popularity_score | volume_factor * freshness_factor |

---

### Examples (Intuition)

| Case | days_since_last_episode | episode_count | Expected popularity_score |
|------|--------------------------|---------------|---------------------------|
| Very active show | 3 | 100 | High |
| Active but small | 7 | 10 | Medium |
| Large but inactive | 180 | 200 | Low |
| Inactive and small | 365 | 5 | Very low |

---

### Implementation Plan (Later)

This is intentionally **not required in v1**.

When implemented, it can be computed via:
- a daily batch job
- or during ingestion updates

Daily recomputation is sufficient for discovery use cases.

---

## Design Decisions

### Why include popularity_score in mapping now?

- Mapping changes are expensive
- Query changes are cheap
- Reserving the field avoids future reindexing

---

### Why keep the index lean?

- Search indexes must stay small and fast
- Canonical podcast metadata belongs elsewhere
- This index exists only for discovery UI

---

## Relationship to Query Template

This mapping is designed to work with the show discovery query:

- weighted multi-field matching (`title > description > publisher`)
- language filtering
- exclusion of documents without `image_url`
- ordering by `_score`, then popularity/activity signals

Mapping and query should always evolve together.