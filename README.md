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

## Milestone 2: local dataset ingestion

Milestone 2 adds a canonical local image-caption registry. It ingests existing canonical CSV
manifests or a directory containing `captions.csv` and referenced images, checks local paths and
supported extensions (`.jpg`, `.jpeg`, `.png`, `.webp`), assigns seeded deterministic splits, and
produces dataset quality statistics without downloading data or loading image/model libraries.

```bash
multimodal-retrieval-ops ingest-local-fixture
multimodal-retrieval-ops split-manifest
multimodal-retrieval-ops inspect-manifest
```

The default workflow reads the tiny tracked fixture in `tests/fixtures/local_dataset`, writes
generated manifests under `data/processed/`, and updates the deterministic tracked report at
`reports/dataset_inspection_report.md`. Split fractions can be configured with
`--train-fraction`, `--validation-fraction`, and `--test-fraction`; the seed defaults to `42`.
