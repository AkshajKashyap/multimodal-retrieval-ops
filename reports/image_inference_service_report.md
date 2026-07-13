# Image Inference Service Report

Run state: **success**

- Backend: `flat`
- Model: `openai/clip-vit-base-patch32`
- Model revision: `default`
- Local files only: `true`
- Retrieval artifacts: `ready`
- Image encoder ready: `true`
- Caption candidates: 5000
- Embedding dimension: 512
- Accepted formats: JPEG, PNG, WEBP
- Maximum upload bytes: 10485760
- Maximum decoded pixels: 20000000
- Detail: in-process arbitrary-image smoke completed successfully

## Smoke result

- Safe image identifier: `04d25b80373cbbf7`
- First request cached: `false`
- Repeated request cached: `true`
- Ranked caption IDs: `test-000181-caption-001, test-000000-caption-003, test-000000-caption-004`

## Input validation and metrics

- Validation: byte size, MIME type, decoded format, corruption, and pixel count
- Arbitrary image requests: 2
- Vision encoder invocations: 1
- Image-cache hits: 1
- Image-cache misses: 1
- Validation errors: 0
- Inference errors: 0
- Uploaded bytes observed: 210866
- Image latency observations: 2

## Limitations

Uploads are bounded, decoded, and processed entirely in memory; they are never saved
to disk. The Hugging Face implementation loads one full CLIPModel object and invokes
only its vision tower for image requests. CPU startup and inference may be slow.
Caches and metrics are process-local. No OCR, augmentation, training, fine-tuning,
reranking, index rebuilding, or new dataset is included.
