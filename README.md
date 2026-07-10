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

## Milestone 3: lexical retrieval baseline

Milestone 3 provides the first retrieval and evaluation backbone. This is a deterministic lexical
bag-of-words baseline—not a multimodal neural model. Its vocabulary is fitted only on train
captions, vectors are L2-normalized, and search uses exact cosine similarity with stable tie
breaking. By default, evaluation uses validation/test captions as queries and limits candidates to
those same held-out splits.

```bash
multimodal-retrieval-ops build-text-baseline-index
multimodal-retrieval-ops search-text-baseline --query "red car"
multimodal-retrieval-ops evaluate-text-baseline
```

The generated index and vocabulary live under `artifacts/baseline/` and remain ignored. The small,
deterministic Markdown and JSON evaluation summaries are tracked under `reports/`. Reported metrics
are Recall@1, Recall@5, Recall@10, MRR, median rank, mean rank, and query count.

## Milestone 4: deterministic multimodal backend

Milestone 4 introduces pluggable text/image encoder interfaces and a shared text-to-image index.
The included backend is a deterministic placeholder—not CLIP and not a neural model. It hashes
text tokens and lightweight local file/path tokens into the same fixed-size normalized space, then
uses exact cosine search. This validates the multimodal architecture without downloads or model
dependencies and must not be interpreted as a model-quality result.

```bash
multimodal-retrieval-ops build-multimodal-baseline-index
multimodal-retrieval-ops search-multimodal-baseline --query "red car"
multimodal-retrieval-ops evaluate-multimodal-baseline
```

The generated `artifacts/baseline/multimodal_index.json` stays ignored. Deterministic evaluation
summaries are written to `reports/multimodal_baseline_report.md` and
`reports/multimodal_baseline_metrics.json`.

## Milestone 5: optional CLIP zero-shot backend

The lightweight default installation remains unchanged and does not install neural dependencies:

```bash
python -m pip install -e ".[dev]"
```

To enable the Hugging Face CLIP backend, install the optional extra:

```bash
python -m pip install -e ".[dev,clip]"
```

CLIP imports and model loading are lazy, so ordinary tests and non-CLIP commands never download a
model. Explicit CLIP model commands may retrieve weights from Hugging Face. Pass
`--local-files-only` when network access must be forbidden.

```bash
multimodal-retrieval-ops clip-backend-info
multimodal-retrieval-ops build-clip-index
multimodal-retrieval-ops search-clip --query "red car"
multimodal-retrieval-ops evaluate-clip
```

The default model is `openai/clip-vit-base-patch32`; override it with `--model-name`. CLIP commands
also accept `--device` (default `cpu`) and `--batch-size`. Generated indexes and embedding caches
remain ignored under `artifacts/clip/`. CLIP results depend on actual model weights, and retrieval
over the tiny fixtures is an integration check—not a model-quality benchmark. No FAISS or
fine-tuning is included yet.

## Milestone 5.5: verified real-model smoke path

After installing the CLIP extra, run the complete CPU smoke workflow with:

```bash
make clip-smoke
```

This loads `openai/clip-vit-base-patch32`, builds the index, searches for `"red car"`, evaluates
held-out retrieval, and builds again to verify a cache hit. The verified local run used 512-element
normalized embeddings for five fixture images. It evaluated two held-out queries and measured
Recall@1/5/10 of `1.0000`, MRR `1.0000`, median rank `1.00`, and mean rank `1.00`. These figures only
show that real neural text/image encoding and retrieval execute end to end on deliberately obvious
tiny images; they are not a meaningful quality benchmark.

## Milestone 6: multi-caption schema and Flickr8k protocol

Schema v2 separates image candidates from caption queries:

- `image_id` identifies an image and may repeat across caption rows.
- `caption_id` identifies one query and must be globally unique.
- Every row for one image must use the same `image_path` and `split`.
- Splitting and seeded subsetting operate on image groups, so captions for one image cannot leak
  across train, validation, and test.
- Image indexes contain one candidate vector per `image_id`; evaluation may issue several caption
  queries whose correct target is that same image.

Legacy manifests with `item_id` remain valid. Migration deterministically maps `item_id` to
`image_id` and creates `<item_id>-caption-001` as the caption identity:

```bash
multimodal-retrieval-ops migrate-manifest-v2
multimodal-retrieval-ops validate-manifest
```

Flickr8k is never downloaded automatically. A typical user-provided local layout is:

```text
/datasets/Flickr8k/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   └── ...
└── captions.txt              # image,caption CSV
```

The original tab-separated `Flickr8k.token.txt` format (`image.jpg#0<TAB>caption`) is also
supported. Ingest and create a deterministic image-group subset with:

```bash
multimodal-retrieval-ops ingest-flickr8k \
  --images-dir /datasets/Flickr8k/Images \
  --captions-file /datasets/Flickr8k/captions.txt
multimodal-retrieval-ops create-benchmark-subset --max-images 1000 --seed 42
multimodal-retrieval-ops evaluate-clip-benchmark \
  --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 16
```

Or run the opt-in workflow using Make variables:

```bash
make clip-benchmark \
  FLICKR8K_IMAGES_DIR=/datasets/Flickr8k/Images \
  FLICKR8K_CAPTIONS_FILE=/datasets/Flickr8k/captions.txt
```

The benchmark compares the lexical, deterministic placeholder, and real zero-shot CLIP paths using
exact cosine search. The earlier five-image fixture metrics remain integration checks; they are
distinct from a real Flickr8k benchmark. This repository had no local Flickr8k copy during
Milestone 6 implementation, so the tracked real-benchmark report is explicitly `not_run`.
