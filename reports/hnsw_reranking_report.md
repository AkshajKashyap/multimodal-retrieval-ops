# HNSW Exact Reranking Report

Run state: **success**

- Model: `openai/clip-vit-base-patch32` (`default`)
- Dataset fingerprint: `f0e3df07c059a4d5516a2f7e9549240844724656ac3217429f65a44f747327ec`
- Manifest fingerprint: `a254e77e91d14559cade632b78329ab4991971feeff5dc218da1917e15bd3d26`
- Source cache fingerprint: `1b44fbfa0c6a48b6ba63affce6b3016c6a4bde7b3301009cb37793813a89109f`
- Split / dimension: `test` / 512
- Images / captions: 1000 / 5000
- Indexes: `IndexFlatIP` and `IndexHNSWFlat`
- Fixed HNSW: M=32, efConstruction=100, efSearch=64
- Fixed exact-reranking shortlist: candidate_k=50
- Artifact compatibility: **passed**
- Rejected adapter embeddings used: **no**

## Text to image

- Queries / candidates: 5000 / 1000
- Flat top-1 in shortlist: 0.9970 (missing 15)
- Mean Flat top-5 / top-10 shortlist fraction: 0.9953 / 0.9931
- Complete Flat top-10 shortlists: 4691 (0.9382)

| Method | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FlatIP | 0.5538 | 0.8160 | 0.8910 | 0.6712 | 1.00 | 5.1948 |
| Raw HNSW | 0.5528 | 0.8144 | 0.8886 | 0.6696 | 1.00 | 21.8230 |
| Reranked HNSW | 0.5528 | 0.8144 | 0.8886 | 0.6696 | 1.00 | 21.8230 |

| Method | Top-1 agreement | Top-5 set | Top-10 set | Mean overlap@5 | Mean overlap@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw HNSW | 0.9970 | 0.9772 | 0.9382 | 4.9764 | 9.9314 |
| Reranked HNSW | 0.9970 | 0.9772 | 0.9382 | 4.9764 | 9.9314 |

- Target-rank changes improved / unchanged / worsened: 0 / 5000 / 0
- Mean / median rank change (positive is improvement): 0.0000 / 0.00
- Largest improvement / regression: 0 / 0
- Maximum exact-score verification difference: 1.788139343e-07

| Timed method | Median batch (s) | Approx. queries/s |
| --- | ---: | ---: |
| Raw HNSW | 0.110631 | 45195.42 |
| Exact shortlist rescoring | 0.185070 | 27016.77 |
| Combined HNSW + reranking | 0.328793 | 15207.14 |
| FlatIP | 0.159402 | 31367.30 |

## Image to text

- Queries / candidates: 1000 / 5000
- Flat top-1 in shortlist: 0.9930 (missing 7)
- Mean Flat top-5 / top-10 shortlist fraction: 0.9916 / 0.9869
- Complete Flat top-10 shortlists: 890 (0.8900)

| Method | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FlatIP | 0.7170 | 0.9170 | 0.9560 | 0.8031 | 1.00 | 2.4980 |
| Raw HNSW | 0.7130 | 0.9160 | 0.9540 | 0.7999 | 1.00 | 27.3710 |
| Reranked HNSW | 0.7130 | 0.9160 | 0.9540 | 0.7999 | 1.00 | 27.3710 |

| Method | Top-1 agreement | Top-5 set | Top-10 set | Mean overlap@5 | Mean overlap@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw HNSW | 0.9930 | 0.9660 | 0.8900 | 4.9580 | 9.8690 |
| Reranked HNSW | 0.9930 | 0.9660 | 0.8900 | 4.9580 | 9.8690 |

- Target-rank changes improved / unchanged / worsened: 0 / 1000 / 0
- Mean / median rank change (positive is improvement): 0.0000 / 0.00
- Largest improvement / regression: 0 / 0
- Maximum exact-score verification difference: 1.788139343e-07

| Timed method | Median batch (s) | Approx. queries/s |
| --- | ---: | ---: |
| Raw HNSW | 0.035878 | 27872.14 |
| Exact shortlist rescoring | 0.046535 | 21489.07 |
| Combined HNSW + reranking | 0.094383 | 10595.11 |
| FlatIP | 0.485120 | 2061.35 |

## Promotion gate

Decision: **fail**

Keep the existing raw FlatIP and HNSW serving behavior.

## Limitations

This is retrieval-backend fidelity evaluation over the already-established official
Flickr8k test artifact, not a new unbiased model-generalization estimate. Rescoring uses
only persisted normalized CLIP vectors; it does not load CLIP or rejected adapters.
Timings use one warmup and three measured whole-query-batch repetitions and are
machine-specific and non-authoritative. FlatIP may remain faster at this dataset scale.
No serving behavior is changed by this offline milestone.
