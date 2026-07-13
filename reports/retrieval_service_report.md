# Retrieval Service Report

Run state: **success**

- Selected backend: `flat`
- Artifact readiness: `ready`
- Detail: in-process cached-ID smoke completed successfully
- Index type: `IndexFlatIP`
- FAISS version: `1.14.3`
- Model: `openai/clip-vit-base-patch32` (`default`)
- Embedding dimension: 512
- Image candidates: 1000
- Caption candidates: 5000
- Split: `test`
- efSearch: not applicable

## Smoke requests

- Health: `alive`
- Ready: `ready`
- Caption-to-image result IDs: `test-000484, test-000375, test-000833`
- Image-to-caption result IDs: `test-000181-caption-001, test-000000-caption-003, test-000000-caption-004`

## Process-local observability

- Total requests observed: 5
- Retrieval latency observations: 2
- Errors: 0
- Unknown query IDs: 0
- Invalid top-k requests: 0

## Limitations

The service accepts cached caption and image IDs only. It does not encode arbitrary
text, accept image uploads, run CLIP, rebuild indexes, or persist metrics. Counters and
latency observations are bounded to one process and reset when that process exits.
FlatIP remains the default correctness-oriented backend; HNSW must be selected
explicitly and is not presented as universally faster or better.
