# FAISS HNSW Comparison Report

Run state: **success**

- Dataset fingerprint: `f0e3df07c059a4d5516a2f7e9549240844724656ac3217429f65a44f747327ec`
- Manifest fingerprint: `a254e77e91d14559cade632b78329ab4991971feeff5dc218da1917e15bd3d26`
- Source cache fingerprint: `1b44fbfa0c6a48b6ba63affce6b3016c6a4bde7b3301009cb37793813a89109f`
- Split: `test`
- Image count: 1000
- Caption count: 5000
- Model: `openai/clip-vit-base-patch32` (`default`)
- Embedding dimension: 512
- FAISS/index: `1.14.3` / `IndexHNSWFlat` / `inner_product`
- Construction: M=32, efConstruction=100
- Bounded search settings: 16, 32, 64

## Text to image

- Queries: 5000
- Candidates: 1000
- FlatIP metrics (R@1/R@5/R@10/MRR/median/mean): 0.5538/0.8160/0.8910/0.6712/1.00/5.19

| Setting | Top-1 agreement | Top-5 ref recall | Top-10 ref recall | Mean overlap@5 | Mean overlap@10 | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| efSearch=16 | 0.9540 | 0.9370 | 0.9168 | 4.685 | 9.168 | 0.5262 | 0.7778 | 0.8494 | 0.6382 | 1.00 | 66.75 |
| efSearch=32 | 0.9866 | 0.9788 | 0.9719 | 4.894 | 9.719 | 0.5460 | 0.8040 | 0.8770 | 0.6611 | 1.00 | 28.26 |
| efSearch=64 | 0.9970 | 0.9953 | 0.9931 | 4.976 | 9.931 | 0.5528 | 0.8144 | 0.8886 | 0.6697 | 1.00 | 9.95 |

| Setting | Absolute ΔR@1 | ΔR@5 | ΔR@10 | ΔMRR | ΔMedian rank | ΔMean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| efSearch=16 | 0.0276 | 0.0382 | 0.0416 | 0.0330 | 0.00 | 61.56 |
| efSearch=32 | 0.0078 | 0.0120 | 0.0140 | 0.0101 | 0.00 | 23.07 |
| efSearch=64 | 0.0010 | 0.0016 | 0.0024 | 0.0015 | 0.00 | 4.76 |

| Setting | Median batch (s) | Mean batch (s) | Approx. queries/s |
| --- | ---: | ---: | ---: |
| FlatIP | 0.255511 | 0.266087 | 18790.84 |
| efSearch=16 | 0.058616 | 0.058424 | 85581.56 |
| efSearch=32 | 0.108242 | 0.105520 | 47384.36 |
| efSearch=64 | 0.186196 | 0.188588 | 26512.77 |

## Image to text

- Queries: 1000
- Candidates: 5000
- FlatIP metrics (R@1/R@5/R@10/MRR/median/mean): 0.7170/0.9170/0.9560/0.8031/1.00/2.50

| Setting | Top-1 agreement | Top-5 ref recall | Top-10 ref recall | Mean overlap@5 | Mean overlap@10 | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| efSearch=16 | 0.9460 | 0.9304 | 0.8986 | 4.652 | 8.986 | 0.6900 | 0.8910 | 0.9280 | 0.7774 | 1.00 | 98.02 |
| efSearch=32 | 0.9790 | 0.9722 | 0.9596 | 4.861 | 9.596 | 0.7080 | 0.9140 | 0.9500 | 0.7955 | 1.00 | 22.67 |
| efSearch=64 | 0.9930 | 0.9916 | 0.9869 | 4.958 | 9.869 | 0.7130 | 0.9160 | 0.9540 | 0.8000 | 1.00 | 12.63 |

| Setting | Absolute ΔR@1 | ΔR@5 | ΔR@10 | ΔMRR | ΔMedian rank | ΔMean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| efSearch=16 | 0.0270 | 0.0260 | 0.0280 | 0.0257 | 0.00 | 95.52 |
| efSearch=32 | 0.0090 | 0.0030 | 0.0060 | 0.0076 | 0.00 | 20.17 |
| efSearch=64 | 0.0040 | 0.0010 | 0.0020 | 0.0031 | 0.00 | 10.13 |

| Setting | Median batch (s) | Mean batch (s) | Approx. queries/s |
| --- | ---: | ---: | ---: |
| FlatIP | 0.591678 | 0.645165 | 1549.99 |
| efSearch=16 | 0.038081 | 0.035586 | 28100.60 |
| efSearch=32 | 0.047893 | 0.049948 | 20020.73 |
| efSearch=64 | 0.063245 | 0.061658 | 16218.61 |

## Recommendation

**HNSW efSearch=64.** Selected the lowest efSearch passing the bounded two-direction accuracy gate.

## Limitations

Timing is machine-specific and non-authoritative. It measures search only, with one
warmup and five measured whole-query-batch repetitions. Mean overlap@k is the mean
intersection count; reference-set recall divides that count by k.
This benchmark has only 1,000 image and 5,000 caption candidates, so FlatIP may be
faster. HNSW is not inherently better at this scale; these results do not establish
production-scale acceleration or model quality.
