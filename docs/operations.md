# Operations

## Install and prepare artifacts

Use `.[dev,faiss,serve]` for complete non-neural verification. `clip` enables Hugging Face CLIP,
`hfdata` enables opt-in Hugging Face dataset ingestion, and `train` enables adapter code. Neural and
dataset extras can download only when explicitly permitted; serving should use local artifacts and
`--local-files-only` for optional live encoders.

Prepare artifacts in this order: canonical schema-v2 manifest, compatible CLIP embedding cache,
FlatIP indexes, then optional HNSW indexes. Validate FlatIP against the exact reference before using
approximate search. The service only loads persisted artifacts and never rebuilds an index.

```bash
multimodal-retrieval-ops serve-retrieval --backend flat
multimodal-retrieval-ops serve-retrieval --backend hnsw --ef-search 64
```

Enable arbitrary text or image inference explicitly with `--enable-text-inference` or
`--enable-image-inference`. Encoder model, revision, backend, and dimension must match the cache.
Text length, upload bytes, pixel count, image formats, top-k, and in-memory cache sizes are bounded.
When local files only is active, unavailable model weights make readiness fail without a download.

`/health` is process liveness and does not prove artifacts are usable. `/ready` reports retrieval
artifact validation and optional encoder states. Do not route retrieval traffic until ready.

## Telemetry and monitoring

Enable telemetry with `--enable-telemetry` and configure a local `.jsonl` path, size, and rotation
count. Events exclude raw text and image bytes. Analyze offline with
`analyze-retrieval-telemetry`; configure minimum observation counts alongside thresholds. Small
samples return `insufficient_data`, not healthy or unhealthy.

Common failures include missing serving/FAISS extras, absent cache/index/manifest files, fingerprint
or model mismatch, wrong split/dimension/candidate ordering, unsupported HNSW settings, unavailable
local model weights, oversized or invalid uploads, invalid top-k, unwritable telemetry paths, and
unsupported telemetry schemas. Repair or rebuild artifacts explicitly outside the service; startup
and readiness intentionally do not mutate them.

The CPU container contains only runtime serving and FAISS dependencies. Mount a compatible
manifest, embedding cache, and index directories at runtime; it intentionally contains no model,
dataset, cache, or real index.
