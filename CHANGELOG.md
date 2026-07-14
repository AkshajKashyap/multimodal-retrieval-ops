# Changelog

All notable changes to this project are documented here.

## [1.0.0]

- Added canonical and schema-v2 multi-caption dataset ingestion with official-split safeguards.
- Added optional zero-shot CLIP embedding, caching, and bidirectional Flickr8k evaluation.
- Added exact FlatIP correctness validation and bounded optional HNSW comparison.
- Added persisted-artifact FastAPI serving for cached IDs and bounded arbitrary text/image input.
- Recorded the validation-only contrastive-adapter experiment and decision to retain zero-shot CLIP.
- Recorded that exact reranking did not improve IndexHNSWFlat rankings and was not promoted.
- Added privacy-safe telemetry, offline monitoring, and minimum-sample health safeguards.
- Added deterministic synthetic portfolio smoke, release consistency validation, documentation,
  CI, and CPU serving-container support.
