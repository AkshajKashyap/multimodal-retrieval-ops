# Contrastive Adapter Training Report

Run state: **success**

## Data and frozen encoder boundaries

- Training split: `train`
- Training subset fingerprint: `48753683fe20a03737a1abb097c55c07d169d3f7f35172ef6ac4b83cbfb29aee`
- Training images/captions: 500 / 2500
- Selection split: `validation`
- Validation subset status: `untouched for gradient updates`
- Validation subset fingerprint: `10c65f8e4b5b50def0ea9ea09c5e3ab8e652cce0ed25b7e11980e1cb95ccd5fa`
- Validation images/captions: 100 / 500
- Official test split accessed: `false`
- Source model: `openai/clip-vit-base-patch32`
- Source model revision: `default`
- Embedding dimension: 512
- CLIP encoder frozen: `true`
- Training input: `cached normalized embeddings only`

## Fixed architecture and configuration

- Architecture: `two-layer-residual-gelu-l2-v1`
- Bottleneck dimension: 128
- Separate image/text parameters: `true`
- Parameter count: 263424
- Seed: 42
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 64 unique images
- Temperature: 0.07
- Maximum epochs: 20
- Early-stopping patience: 4
- Selected epoch: 6
- Early stopped: `true`
- Selection metric: `validation mean bidirectional MRR`

## Training history

| Epoch | Training loss | Validation mean bidirectional MRR |
| ---: | ---: | ---: |
| 1 | 2.053616 | 0.892378 |
| 2 | 1.564032 | 0.872376 |
| 3 | 1.229863 | 0.873941 |
| 4 | 0.921216 | 0.893435 |
| 5 | 0.721982 | 0.885002 |
| 6 | 0.638473 | 0.898587 |
| 7 | 0.551485 | 0.886593 |
| 8 | 0.499063 | 0.895336 |
| 9 | 0.455785 | 0.881174 |
| 10 | 0.431466 | 0.891631 |

## Validation retrieval metrics

| Representation and direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank | Queries | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot text to image | 0.8100 | 0.9640 | 0.9820 | 0.8773 | 1.00 | 1.70 | 500 | 100 |
| Zero-shot image to text | 0.9200 | 0.9700 | 1.0000 | 0.9437 | 1.00 | 1.25 | 100 | 500 |
| Adapted text to image | 0.7800 | 0.9460 | 0.9860 | 0.8538 | 1.00 | 1.74 | 500 | 100 |
| Adapted image to text | 0.9000 | 0.9900 | 1.0000 | 0.9433 | 1.00 | 1.16 | 100 | 500 |

| Absolute adapted-minus-zero-shot difference | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text to image | -0.0300 | -0.0180 | +0.0040 | -0.0235 | +0.00 | +0.03 |
| Image to text | -0.0200 | +0.0200 | +0.0000 | -0.0003 | +0.00 | -0.09 |

## Limitations

This is one bounded adapter configuration over a small training subset. CLIP was
not fine-tuned, and the official test split was not inspected. Validation selected
the checkpoint and therefore is not an unbiased final benchmark. No hyperparameter
search, reranking, LoRA, full-model gradients, or production-quality claim is included.
A negative promotion result is expected and acceptable.
