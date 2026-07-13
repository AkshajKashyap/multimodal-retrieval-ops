# Text Inference Service Report

Run state: **success**

- Backend: `flat`
- Model: `openai/clip-vit-base-patch32`
- Model revision: `default`
- Local files only: `true`
- Retrieval artifacts: `ready`
- Text encoder ready: `true`
- Image candidates: 1000
- Embedding dimension: 512
- Detail: in-process arbitrary-text smoke completed successfully

## Smoke query

- Query: `a dog running outside`
- First request cached: `false`
- Repeated request cached: `true`
- Ranked image IDs: `test-000412, test-000954, test-000243`

## Process-local metrics

- Arbitrary text requests: 2
- Text encoder invocations: 1
- Query-cache hits: 1
- Query-cache misses: 1
- Text inference errors: 0
- Text latency observations: 2

## Limitations

The Hugging Face implementation loads the full CLIPModel object because text and
vision projections share that model package, but this workflow invokes only the text
tower. It never decodes or embeds images. Queries run on CPU by default and may be
slow. The cache and metrics are bounded, process-local, and not persisted.
No image upload, training, fine-tuning, reranking, or index rebuilding is included.
