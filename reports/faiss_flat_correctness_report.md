# FAISS Flat Correctness Report

Run state: **success**

- FAISS version: `1.14.3`
- Index type: `IndexFlatIP`
- Source cache fingerprint: `1b44fbfa0c6a48b6ba63affce6b3016c6a4bde7b3301009cb37793813a89109f`
- Dataset fingerprint: `f0e3df07c059a4d5516a2f7e9549240844724656ac3217429f65a44f747327ec`
- Manifest fingerprint: `a254e77e91d14559cade632b78329ab4991971feeff5dc218da1917e15bd3d26`
- Split: `test`
- Model: `openai/clip-vit-base-patch32`
- Model revision: `default`
- Embedding dimension: 512
- Image count: 1000
- Caption count: 5000

## Text to image

- Queries: 5000
- Candidates: 1000
- Correctness gate: **pass**
- Top-1 agreement: 1.000000
- Top-5 set agreement: 1.000000
- Top-10 set agreement: 1.000000
- Maximum score difference: 0.0000002980
- Tie-explained disagreements (1/5/10): 0/0/0

| Backend | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 0.5538 | 0.8160 | 0.8910 | 0.6712 | 1.00 | 5.19 |
| FAISS FlatIP | 0.5538 | 0.8160 | 0.8910 | 0.6712 | 1.00 | 5.19 |

## Image to text

- Queries: 1000
- Candidates: 5000
- Correctness gate: **pass**
- Top-1 agreement: 1.000000
- Top-5 set agreement: 1.000000
- Top-10 set agreement: 1.000000
- Maximum score difference: 0.0000002980
- Tie-explained disagreements (1/5/10): 0/0/0

| Backend | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 0.7170 | 0.9170 | 0.9560 | 0.8031 | 1.00 | 2.50 |
| FAISS FlatIP | 0.7170 | 0.9170 | 0.9560 | 0.8031 | 1.00 | 2.50 |

## Tie handling and limitations

Set disagreements are classified as tie-explained only when every differing candidate
is within the score tolerance of the reference top-k boundary.
IndexFlatIP is exact; this validates retrieval-backend correctness, not model quality.
No latency claims, approximate indexes, GPU FAISS, or neural inference are included.
