# Elasticsearch Mapping – Episodes Index

This document defines the **current Elasticsearch index mapping**
for **podcast episodes**, updated to exactly reflect the latest
JSON mapping configuration.

The mapping is optimized for **intent-driven episode search** with
strict schema control, autocomplete support, and predictable
performance.

---

## Index Purpose

The `episodes` index supports **episode-level search and retrieval**.

Typical user intent:
- Search for a specific topic or discussion
- Find a particular episode
- Browse episodes within a show

Design priorities:
- Precision over recall
- Fast highlighting on text fields
- Stable sorting and pagination
- Strict schema enforcement

---

## Index-Level Settings

### Analyzers & Filters

The index defines a custom autocomplete analyzer using edge n-grams.

- **edge_ngram_filter**
  - `min_gram`: 2
  - `max_gram`: 20

- **autocomplete_analyzer**
  - Tokenizer: `standard`
  - Filters: `lowercase`, `edge_ngram_filter`

- **autocomplete_search**
  - Tokenizer: `standard`
  - Filters: `lowercase`

These analyzers are used exclusively for type-ahead search on titles.

---

## Dynamic Mapping Rules

### Global Dynamic Mode

```
dynamic: strict
```

All fields must be explicitly defined.
Unexpected fields will be rejected at index time.

---

### Dynamic Templates

#### External IDs (keyword, indexed)

Applies to:
- `external_ids.*`
- `show.external_ids.*`

```
type: keyword
ignore_above: 256
```

Rules:
- Keys represent external sources
- Values must be strings
- One ID per source
- No nested objects or arrays

---

#### External URLs (keyword, not indexed)

Applies to:
- `external_urls.*`

```
type: keyword
index: false
ignore_above: 512
```

Used for UI or outbound linking only.

---

## Core Episode Fields

| Field | Type | Indexed | Required | Description |
|------|------|---------|----------|-------------|
| episode_id | keyword | Yes | Yes | Internal stable episode identifier |
| external_ids | object | Yes | No | External ID map (dynamic keys) |
| external_urls | object | No | No | External URLs (UI only) |
| title | text | Yes | Yes | Episode title |
| description | text | Yes | No | Episode description |
| language | keyword | Yes | Yes | Normalized language code |
| published_at | date | Yes | Yes | Publish datetime |
| duration_sec | integer | Yes | No | Episode duration (seconds) |
| image_url | keyword | No | No | Episode image URL (UI only) |

---

## Title Subfields

| Subfield | Type | Purpose |
|--------|------|--------|
| title.autocomplete | text | Prefix / type-ahead search |
| title.keyword | keyword | Exact match and sorting |

---

## Audio Object

Episodes include a strict `audio` object.

```
audio (object, dynamic: strict)
```

| Field | Type | Indexed | Description |
|------|------|---------|-------------|
| audio.url | keyword | No | Audio file URL |
| audio.type | keyword | Yes | MIME / audio type |
| audio.length_bytes | long | Yes | File size in bytes |

---

## Embedded Show Snapshot

Episodes store a **denormalized snapshot** of show metadata under
the `show` field.

```
show (object, dynamic: strict)
```

| Field | Type | Indexed | Description |
|------|------|---------|-------------|
| show.show_id | keyword | Yes | Internal show identifier |
| show.external_ids | object | Yes | External ID map |
| show.title | text | Yes | Show title |
| show.publisher | text | Yes | Show publisher |
| show.image_url | keyword | No | Show image (UI only) |

---

## Timestamps

| Field | Type | Description |
|------|------|-------------|
| created_at | date | Record creation time |
| updated_at | date | Last update time |

---

## Summary

The `episodes` index mapping is strict, autocomplete-enabled,
and optimized for fast, precise search.
