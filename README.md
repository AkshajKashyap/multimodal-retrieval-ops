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

## Milestone 6.5: Hugging Face Flickr8k bidirectional benchmark

The real benchmark is opt-in and keeps the default install and `make check` download-free:

```bash
python -m pip install -e ".[dev,clip,hfdata]"
multimodal-retrieval-ops ingest-hf-flickr8k --dataset-name jxie/flickr8k
multimodal-retrieval-ops inspect-hf-flickr8k
```

Images are materialized under ignored `data/raw/hf_flickr8k/images/<split>/` directories with
deterministic filenames and safe restart behavior. The source-provided train, validation, and test
splits are preserved exactly; ingestion never resplits them.

Run the 100-image official-test integration benchmark with:

```bash
multimodal-retrieval-ops evaluate-clip-flickr8k \
  --split test --max-images 100 \
  --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 16
# equivalent opt-in target
make clip-flickr8k-smoke
```

Run all 1,000 official test images and 5,000 captions with:

```bash
multimodal-retrieval-ops evaluate-clip-flickr8k \
  --split test --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 16
# equivalent opt-in target
make clip-flickr8k-benchmark
```

Text-to-image uses each caption as a query against one candidate per image. Image-to-text uses each
image as a query against all captions; all five captions belonging to that image are relevant, and
the rank is the highest-ranked relevant caption. Both directions use vectorized exact cosine
similarity.

The measured CPU run with `openai/clip-vit-base-patch32` produced:

| Mode and direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100-image integration, text to image | 0.8060 | 0.9780 | 0.9920 | 0.8811 | 1.00 | 1.45 |
| 100-image integration, image to text | 0.9300 | 1.0000 | 1.0000 | 0.9583 | 1.00 | 1.12 |
| Full test, text to image | 0.5538 | 0.8160 | 0.8910 | 0.6712 | 1.00 | 5.19 |
| Full test, image to text | 0.7170 | 0.9170 | 0.9560 | 0.8031 | 1.00 | 2.50 |

Dataset source: `jxie/flickr8k`. Its licensing information is not clearly exposed by the source,
so licensing status remains **unresolved**. Downloaded images and Hugging Face caches must not be
redistributed or committed through this repository.

## Milestone 7A: exact FAISS FlatIP correctness

Install the optional CPU-only FAISS backend separately from CLIP and dataset dependencies:

```bash
python -m pip install -e ".[dev,faiss]"
```

Milestone 7A reuses the existing official-test embedding cache. It never loads CLIP, runs model
inference, or downloads data. Build two exact indexes and compare them with the NumPy cosine
reference using:

```bash
multimodal-retrieval-ops faiss-backend-info
multimodal-retrieval-ops build-faiss-flat-indexes
multimodal-retrieval-ops evaluate-faiss-flat
```

Searches accept cached query identities rather than new text or image inputs:

```bash
multimodal-retrieval-ops search-faiss-text \
  --query-caption-id test-000000-caption-001 --k 10
multimodal-retrieval-ops search-faiss-image \
  --query-image-id test-000000 --k 10
```

`faiss.IndexFlatIP` performs exhaustive inner-product search. Because the cached CLIP vectors are
L2-normalized, inner product equals cosine similarity. This is still exact retrieval; approximate
HNSW, IVF, PQ, GPU indexes, and parameter tuning are deliberately deferred.

The measured Flickr8k comparison reproduced all text-to-image and image-to-text retrieval metrics
exactly, with 100% top-1/top-5/top-10 agreement and a maximum score difference of approximately
`2.98e-7`. This validates retrieval-backend correctness only—it does not improve or reevaluate
model quality. Generated binary indexes and companion metadata remain ignored under
`artifacts/faiss/`.

## Milestone 7B: bounded FAISS HNSW comparison

HNSW is an approximate graph index: `M` controls graph connectivity, `efConstruction` controls
work while building the graph, and `efSearch` controls the search-time accuracy/work trade-off.
This milestone fixes construction to `IndexHNSWFlat` with inner product, `M=32`, and
`efConstruction=100`. It compares exactly three settings: `efSearch=16`, `32`, and `64`.

The workflow reuses the existing 1,000-image/5,000-caption official-test embedding cache and the
Milestone 7A FlatIP indexes. It does not load CLIP, regenerate embeddings, ingest data, or download
anything:

```bash
python -m pip install -e ".[dev,faiss]"
multimodal-retrieval-ops build-faiss-hnsw-indexes
multimodal-retrieval-ops evaluate-faiss-hnsw
multimodal-retrieval-ops search-hnsw-text \
  --query-caption-id test-000000-caption-001 --ef-search 32
multimodal-retrieval-ops search-hnsw-image \
  --query-image-id test-000000 --ef-search 32
```

Use `make faiss-hnsw-check` for focused synthetic tests and `make faiss-hnsw-eval` for the bounded
real comparison. FlatIP remains the exhaustive correctness reference; HNSW results quantify
approximation agreement, downstream retrieval metrics, and machine-specific search-only timing.
At only 1,000 or 5,000 candidates, FlatIP may be faster. These measurements do not establish
production-scale acceleration, and approximate search is not automatically preferable.

On the recorded local run, only `efSearch=64` passed the conservative two-direction gate. Its
text-to-image absolute differences from FlatIP were `0.0024` for Recall@10 and `0.0015` for MRR,
with `0.9931` top-10 reference-set recall. Image-to-text differences were `0.0020` and `0.0031`,
with `0.9869` top-10 reference-set recall. Timing is reported for transparency in the tracked
report, but remains machine-specific and non-authoritative.

## Milestone 7C: local persisted-index retrieval service

Install the optional HTTP service and FAISS dependencies without installing CLIP:

```bash
python -m pip install -e ".[dev,faiss,serve]"
```

The service requires the existing official-test embedding cache, schema-v2 manifest, and both
persisted indexes for the selected backend. Startup validates their model metadata, dimensions,
fingerprints, candidate ordering, index type, and source-cache identity. Missing or stale artifacts
leave the process live but unready; they are never rebuilt automatically.

Inspect or smoke-test the default FlatIP service without opening a network port:

```bash
multimodal-retrieval-ops retrieval-service-info --backend flat
multimodal-retrieval-ops retrieval-service-smoke --backend flat
```

Start a local server explicitly when needed:

```bash
multimodal-retrieval-ops serve-retrieval --backend flat
multimodal-retrieval-ops serve-retrieval --backend hnsw --ef-search 64
```

FlatIP remains the default correctness-oriented backend. HNSW must be selected explicitly and uses
only the bounded supported `efSearch` values; `64` is the Milestone 7B recommendation for these
artifacts, not a universal performance claim.

The endpoints are:

- `GET /health` for process liveness only.
- `GET /ready` for persisted-artifact readiness and concise failure reasons.
- `GET /index-info` for backend, model, dimension, count, and fingerprint metadata.
- `POST /retrieve/images` for cached `caption_id` to ranked image retrieval.
- `POST /retrieve/captions` for cached `image_id` to ranked caption retrieval.
- `GET /metrics` for bounded process-local counters and latency summaries.

Only IDs already represented in the embedding cache are accepted. Arbitrary user text encoding,
image uploads, and new image inference are intentionally deferred. Metrics are neither durable nor
distributed: they reset on process restart and describe only one process.

## Milestone 8A: bounded arbitrary text-to-image inference

Install the existing optional CLIP, FAISS, and service extras:

```bash
python -m pip install -e ".[dev,clip,faiss,serve]"
```

Arbitrary text inference is disabled by default. Cached-ID retrieval therefore continues to start
without loading CLIP. Enabling text search requires the existing 1,000-image persisted index and
embedding cache plus locally cached `openai/clip-vit-base-patch32` weights. Model loading uses
local-files-only behavior unless explicitly overridden:

```bash
multimodal-retrieval-ops retrieval-service-info \
  --backend flat --enable-text-inference --local-files-only
multimodal-retrieval-ops serve-retrieval \
  --backend flat --enable-text-inference --local-files-only
```

Search through the API:

```bash
curl -X POST http://127.0.0.1:8000/search/text \
  -H 'content-type: application/json' \
  -d '{"query":"a dog running outside","top_k":5}'
```

Or run one local in-process query without binding a port:

```bash
multimodal-retrieval-ops search-live-text \
  --backend flat --query "a dog running outside" --top-k 5 --local-files-only
```

The service normalizes query strings and keeps a bounded process-local LRU embedding cache, so a
repeated equivalent query avoids another encoder call. The Hugging Face implementation must load
the full `CLIPModel` object, but it invokes only the text tower—there is no image decoding or vision
inference in this milestone. Query vectors are checked for finite values, dimension compatibility,
and L2 normalization before FlatIP or optional HNSW search. FlatIP remains the conservative default;
HNSW is enabled explicitly with `--backend hnsw --ef-search 64`.

CPU model startup and uncached queries may be slow. The cache and metrics are local to one process
and reset at shutdown. Image uploads, image inference, fine-tuning, training, and reranking remain
intentionally deferred.

## Milestone 8B: bounded arbitrary image-to-text inference

Install the CLIP, FAISS, and serving extras; multipart support and Pillow are included in the
serving extra:

```bash
python -m pip install -e ".[dev,clip,faiss,serve]"
```

Arbitrary image inference is disabled by default. Enabling it requires the existing persisted
official-test caption index with 5,000 candidates, its compatible embedding cache and manifest,
and locally cached `openai/clip-vit-base-patch32` files. Startup validates that the model name,
revision, backend, and vision projection dimension match those artifacts. It never downloads model
files or rebuilds embeddings or indexes when `--local-files-only` is used, which is the default.

Inspect readiness, run one bounded in-process smoke, or start the local service with:

```bash
multimodal-retrieval-ops retrieval-service-info \
  --backend flat --enable-image-inference --local-files-only
multimodal-retrieval-ops retrieval-service-smoke \
  --backend flat --enable-image-inference --local-files-only
multimodal-retrieval-ops serve-retrieval \
  --backend flat --enable-image-inference --local-files-only
```

Submit a multipart image through the API:

```bash
curl -X POST http://127.0.0.1:8000/search/image \
  -F 'image=@query.jpg;type=image/jpeg' \
  -F 'top_k=5'
```

Or execute one local image query without binding a network port:

```bash
multimodal-retrieval-ops search-live-image \
  --backend flat --image-path query.jpg --top-k 5 --local-files-only
```

JPEG, PNG, and WEBP are accepted after MIME and decoded-format validation. Defaults limit uploads
to 10 MiB and decoded images to 20 million pixels; empty, oversized, corrupt, mismatched, and
decompression-bomb inputs are rejected. Uploaded bytes are parsed and decoded in memory and are
never stored. A bounded process-local LRU keyed by the image SHA-256 and model identity avoids a
second vision pass for identical bytes. When text and image inference are both enabled with the
same configuration, the text and vision towers share one loaded CLIP model object.

Cached-ID retrieval, arbitrary text-to-image search, and arbitrary image-to-caption search can
coexist. Optional HNSW caption search uses `--backend hnsw --ef-search 64`; FlatIP remains the
default. CPU model startup and vision inference may be slow and memory-intensive. Caches and
metrics are process-local. Fine-tuning, training, reranking, OCR, and image persistence remain
deferred.

## Milestone 9A: bounded frozen-embedding contrastive adapters

Milestone 9A is one controlled learning experiment, not full CLIP fine-tuning. It selects at most
500 official training images and 100 separate official validation images, preserving all five
captions for every selected image. Selection is deterministic by image group with seed `42`. The
official test split is rejected by the adapter protocol and remains untouched for a later
milestone.

Install the optional training dependency and, for real frozen-embedding preparation, the existing
CLIP extra:

```bash
python -m pip install -e ".[dev,train]"
python -m pip install -e ".[dev,clip,train]"
```

Preparation uses only materialized Flickr8k files and locally cached
`openai/clip-vit-base-patch32` weights. It creates each compatible train or validation cache once
and reuses it thereafter:

```bash
multimodal-retrieval-ops prepare-adapter-embeddings \
  --train-images 500 --validation-images 100 --seed 42 \
  --model-name openai/clip-vit-base-patch32 --device cpu --local-files-only
multimodal-retrieval-ops train-contrastive-adapters \
  --seed 42 --device cpu --max-epochs 20 \
  --early-stopping-patience 4 --batch-size 64
multimodal-retrieval-ops evaluate-contrastive-adapters --device cpu
multimodal-retrieval-ops contrastive-adapter-info
```

The single architecture uses separate text and image adapters:
`normalize(input + W2(GELU(W1(input))))`, with a 512-dimensional input and output and a
128-dimensional bottleneck. CLIP is always frozen; training consumes cached normalized embeddings
and never reopens images or invokes CLIP. The fixed configuration is learning rate `1e-3`, weight
decay `1e-4`, temperature `0.07`, at most 20 epochs, batch size 64 unique images, and early-stopping
patience 4. A multi-positive symmetric loss treats every caption belonging to an image as relevant,
avoiding false negatives among sibling captions. Checkpoint selection uses validation mean
bidirectional MRR.

Promotion is deliberately conservative: mean bidirectional MRR must improve by at least `0.005`,
while neither direction's Recall@10 nor MRR may fall by more than `0.005`. Otherwise the report
recommends retaining zero-shot CLIP. The small subset may fail to improve the zero-shot model, and
either outcome is an experiment result rather than a quality claim. No test evaluation, CLIP
gradients, LoRA, hyperparameter sweep, reranking, or index rebuilding is included.

The recorded bounded CPU run stopped after 10 epochs and selected epoch 6. On the 100-image/500-
caption validation subset, zero-shot versus adapted MRR was `0.8773` versus `0.8538` for text to
image and `0.9437` versus `0.9433` for image to text. Mean bidirectional MRR changed by `-0.011897`,
and text-to-image MRR exceeded the allowed regression. The conservative decision is therefore to
retain zero-shot CLIP. These validation results selected the checkpoint and are not a final test
benchmark.

## Milestone 9B: validation-only adapter failure analysis

Milestone 9B explains the rejected 9A adapter without running CLIP, regenerating embeddings,
retraining, changing hyperparameters, or inspecting the official test split. It reads only the
existing frozen train/validation caches, selected checkpoint, training history, tracked 9A metrics,
and validation caption metadata:

```bash
multimodal-retrieval-ops contrastive-adapter-diagnostics-info
multimodal-retrieval-ops analyze-contrastive-adapter
# focused synthetic checks
make contrastive-adapter-diagnostics-check
```

The analyzer first verifies model, revision, dimensions, subset and file fingerprints, selected
IDs, image-caption relationships, architecture, and parameter count. Reproduced zero-shot and
adapted metrics must match the tracked 9A values within `1e-6` or analysis stops without overwriting
the earlier decision.

On the existing 100-image/500-caption validation subset, text-to-image ranks improved for 47
queries, were unchanged for 385, and worsened for 68. Image-to-text ranks improved for 7 images,
were unchanged for 85, and worsened for 8. Mean adapted-minus-zero-shot rank change was `+0.032`
for text to image and `-0.090` for image to text. Although mean margins increased, positive-margin
coverage fell from `81%` to `78%` for text queries and from `92%` to `90%` for image queries.

Original-to-adapted cosine similarity averaged `0.6858` for images and `0.6443` for text, showing
more aggressive text movement. Training loss continued down after selected epoch 6 while all four
later validation scores remained below the selected score. The supported diagnosis is likely
overfitting, likely optimization imbalance between modalities, representation movement that was
too aggressive, and insufficient improvement signal. These are bounded descriptive findings, not
proof of causality.

The decision remains **retain zero-shot CLIP**. No second configuration was trained, no serving
index was changed, and the official test split remains untouched.

## Milestone 10A: bounded exact reranking

Milestone 10A adds an offline two-stage retrieval comparison over the persisted official-test
Flickr8k artifacts. HNSW generates exactly 50 candidates with the existing `M=32`,
`efConstruction=100`, and `efSearch=64` configuration. The candidates are then reranked by exact
inner product against their original cached, finite, L2-normalized CLIP vectors. Original HNSW
shortlist order breaks exact-score ties deterministically.

Install the optional CPU FAISS dependency, run focused checks, and execute the one fixed protocol:

```bash
python -m pip install -e ".[dev,faiss]"
make hnsw-reranking-check
multimodal-retrieval-ops evaluate-hnsw-reranking --candidate-k 50 --ef-search 64
multimodal-retrieval-ops hnsw-reranking-info
```

Cached-ID searches exercise the same persisted two-stage path without loading CLIP:

```bash
multimodal-retrieval-ops search-reranked-text --query-caption-id <CAPTION_ID>
multimodal-retrieval-ops search-reranked-image --query-image-id <IMAGE_ID>
```

The evaluation compares FlatIP, raw HNSW, and exactly reranked HNSW in both directions, including
shortlist coverage, end-task metrics, agreement, rank changes, exact-score verification, and a
conservative promotion gate. It measures retrieval-backend fidelity over an established artifact;
it is not a new model-quality estimate. The rejected contrastive adapters are never loaded or used,
and no CLIP inference, embedding generation, training, index rebuilding, API, or serving changes are
part of this milestone. Timing is machine-specific, and reranking may not improve latency at
Flickr8k scale.

## Milestone 11A: privacy-safe retrieval monitoring

Milestone 11A adds optional versioned JSONL telemetry to the existing retrieval service and a
standard-library offline analyzer. Telemetry is disabled by default and does not alter retrieval
ranking, indexes, model behavior, or existing response contracts. No external observability service,
database, collector, or persistent server is required.

Enable bounded local telemetry on an existing service command with:

```bash
multimodal-retrieval-ops serve-retrieval \
  --backend flat \
  --enable-telemetry \
  --telemetry-path logs/retrieval_telemetry.jsonl \
  --telemetry-max-bytes 5242880 \
  --telemetry-backup-count 2
```

The sink rotates before the configured byte limit is exceeded, keeps at most the configured number
of siblings, and flushes each event by default. Write failures never fail retrieval requests and are
counted in process-local `/metrics`. Runtime JSONL files and rotations are ignored by Git.

Telemetry stores only derived operational fields and SHA-256 query identities. It intentionally
excludes raw arbitrary text, caption text, image bytes, uploaded filenames, absolute paths, request
and authentication headers, model-cache paths, stack traces, and complete ranked payloads. Inspect,
smoke, and analyze it without binding a network port:

```bash
multimodal-retrieval-ops retrieval-telemetry-info
multimodal-retrieval-ops retrieval-telemetry-smoke --backend flat
multimodal-retrieval-ops analyze-retrieval-telemetry
```

Default health thresholds allow at most a `0.25` error rate and zero readiness failures, require
labeled Recall@10 of at least `0.80` and labeled MRR of at least `0.50`, and leave p95 latency
unbounded unless configured. Minimum sample safeguards require 20 request observations for error
rate, 20 latency observations when a p95 limit is enabled, 50 labeled cached-ID queries for each
Recall@10 and MRR check, and one readiness observation. These conservative minimums prevent a tiny
smoke window from being misrepresented as production-health evidence.

Each enabled check independently reports `pass`, `fail`, or `insufficient_data`; an unset optional
threshold reports `disabled`. The overall decision is `warning` when any sufficiently sampled check
fails, `healthy` only when every enabled check has enough data and passes, and `insufficient_data`
when nothing sufficiently sampled fails but at least one enabled check lacks evidence. Known-label
quality uses only cached-ID queries with existing manifest relevance; arbitrary text and uploaded-
image queries remain explicitly unlabeled.

The five-event smoke is therefore expected to return `insufficient_data`: it validates telemetry
collection, privacy, serialization, and offline analysis, but cannot support a production-health
conclusion. Controlled synthetic tests can lower minimums explicitly without changing defaults:

```bash
multimodal-retrieval-ops analyze-retrieval-telemetry \
  --min-error-rate-observations 5 \
  --min-latency-observations 5 \
  --min-labeled-recall-observations 2 \
  --min-labeled-mrr-observations 2 \
  --min-readiness-observations 1
```

Monitoring is process-local, single-instance, and single-window. Local rotation is not durable
multi-instance observability, and score margins are operational signals rather than calibrated model
confidence. This milestone does not change retrieval rankings or serving endpoints.
