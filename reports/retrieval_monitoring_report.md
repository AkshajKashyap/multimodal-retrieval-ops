# Retrieval Monitoring Report

Run state: **success**

- Telemetry schema version: 1
- Event count: 5
- Source files: 1

## Traffic and reliability

- Requests by endpoint: `{"cached_caption_to_image": 2, "cached_image_to_caption": 1, "health": 1, "readiness": 1}`
- Requests by backend: `{"flat": 5}`
- Requests by direction: `{"caption_to_image": 2, "image_to_caption": 1, "operational": 2}`
- Successful / failed: 4 / 1
- Overall error rate: 0.2000
- HTTP statuses: `{"200": 4, "404": 1}`
- Error categories: `{"unknown_cached_id": 1}`
- Readiness failures: 0
- Telemetry parsing errors: 0

## Latency and cache

- Latency observations: 5
- Mean / median / p95 / maximum milliseconds: 1.3730 / 0.8971 / 3.1518 / 3.1518
- Cache hits / misses / hit rate: 0 / 0 / unavailable

## Score confidence

- Top-1 score observations: 2
- Mean / median top-1 score: 0.3281 / 0.3281
- Mean / median top-1-to-top-2 margin: 0.0138 / 0.0138
- Non-positive margin rate: 0.0000

## Known-label quality

- Labeled cached-ID queries: 2
- Mean Recall@1 / @5 / @10: 0.0000 / 0.5000 / 0.5000
- Mean reciprocal rank: 0.5000

Known-label quality applies only to cached-ID queries. Arbitrary text and uploaded
image queries are explicitly unlabeled; no relevance judgments are invented.

## Health decision

Decision: **warning**

- maximum_error_rate: observed=0.2000, threshold=0.2500, passed=True
- maximum_readiness_failures: observed=0, threshold=0, passed=True
- minimum_labeled_recall_at_10: observed=0.5000, threshold=0.8000, passed=False
- minimum_labeled_mrr: observed=0.5000, threshold=0.5000, passed=True

## Privacy and operational limitations

Telemetry excludes raw text, caption text, image bytes, uploaded filenames, absolute
paths, headers, model-cache paths, stack traces, and ranked payloads. Query identity is
represented only by SHA-256. Reports exclude timestamps, event IDs, query hashes, and
raw JSONL records. Monitoring is process-local and single-window; local file rotation
is not a durable or multi-instance observability system. Telemetry write failures do
not fail retrieval and are visible only in process-local metrics. Retrieval rankings,
models, indexes, and API response contracts are unchanged.
