# Multimodal Retrieval Ops

Production-style image-text retrieval system with CLIP-style embeddings, contrastive fine-tuning, FAISS search, reranking, evaluation, FastAPI serving, and monitoring.

## Goal

Build an end-to-end multimodal retrieval system:

data ingestion -> embedding generation -> contrastive fine-tuning -> vector index -> retrieval evaluation -> API serving -> telemetry -> monitoring.

## Milestone 1: project foundation

Milestone 1 provides a lightweight, CPU-only Python package for deterministic demo manifest
generation, validation, and reporting. A manifest is a CSV of image-caption pairs with stable
`item_id`, `image_path`, `caption`, `split`, and `source` fields. No images or models are downloaded.

```bash
python -m pip install -e ".[dev]"
multimodal-retrieval-ops --version
multimodal-retrieval-ops project-info
multimodal-retrieval-ops generate-demo-manifest
multimodal-retrieval-ops validate-manifest
multimodal-retrieval-ops generate-manifest-report
```

The default manifest is written to `data/processed/demo_manifest.csv` (ignored as generated data),
and its deterministic summary is written to `reports/demo_manifest_summary.md`.

Custom paths are supported with `--output` and `--manifest`; run any command with `--help` for
details. Run all local quality checks with `make check`.
