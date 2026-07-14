# Contrastive Adapter Failure Analysis

Run state: **success**

## Artifact compatibility

- Compatibility: `passed`
- Model: `openai/clip-vit-base-patch32`
- Model revision: `default`
- Embedding dimension: 512
- Validation subset fingerprint: `10c65f8e4b5b50def0ea9ea09c5e3ab8e652cce0ed25b7e11980e1cb95ccd5fa`
- Validation images/captions: 100 / 500
- Checkpoint architecture: `two-layer-residual-gelu-l2-v1`
- Adapter parameters: 263424
- Cache, checkpoint, selected-ID, relationship, and recorded-metric checks: `passed`
- Official test split accessed: `false`

## Reproduced Milestone 9A validation metrics

All reproduced values matched the tracked Milestone 9A metrics within `1e-6`.

| Representation and direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank | Queries | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot text to image | 0.8100 | 0.9640 | 0.9820 | 0.8773 | 1.00 | 1.70 | 500 | 100 |
| Zero-shot image to text | 0.9200 | 0.9700 | 1.0000 | 0.9437 | 1.00 | 1.25 | 100 | 500 |
| Adapted text to image | 0.7800 | 0.9460 | 0.9860 | 0.8538 | 1.00 | 1.74 | 500 | 100 |
| Adapted image to text | 0.9000 | 0.9900 | 1.0000 | 0.9433 | 1.00 | 1.16 | 100 | 500 |

## Per-query rank outcomes

### Text to image

- Queries: 500
- Improved: 47 (9.40%)
- Unchanged: 385 (77.00%)
- Worsened: 68 (13.60%)
- Mean adapted-minus-zero-shot rank change: +0.0320
- Median rank change: +0.00
- Largest improvement: -39 ranks
- Largest regression: +11 ranks

### Image to text

- Queries: 100
- Improved: 7 (7.00%)
- Unchanged: 85 (85.00%)
- Worsened: 8 (8.00%)
- Mean adapted-minus-zero-shot rank change: -0.0900
- Median rank change: +0.00
- Largest improvement: -5 ranks
- Largest regression: +3 ranks

## Positive-versus-negative similarity margins

For text queries, relevant similarity is the target image score. For image queries,
it is the highest score among that image's relevant captions; irrelevant similarity
is the highest score outside the relevant set.

### Text to image

- Zero-shot mean / median margin: +0.033237 / +0.035412
- Adapted mean / median margin: +0.070114 / +0.074991
- Positive-margin queries, zero-shot / adapted: 81.00% / 78.00%
- Margin improved / worsened: 72.40% / 27.60%
- Largest margin gain / loss: +0.317213 / -0.177124

### Image to text

- Zero-shot mean / median margin: +0.039622 / +0.037994
- Adapted mean / median margin: +0.080355 / +0.069152
- Positive-margin queries, zero-shot / adapted: 92.00% / 90.00%
- Margin improved / worsened: 78.00% / 22.00%
- Largest margin gain / loss: +0.193510 / -0.065739

## Representation movement

Cosine similarity compares each original frozen embedding with its adapted value.

| Modality | Count | Mean | Median | Minimum | Maximum | P5 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Image | 100 | 0.685808 | 0.704766 | 0.494735 | 0.822715 | 0.533159 | 0.788668 |
| Text | 500 | 0.644330 | 0.642891 | 0.434516 | 0.847987 | 0.546825 | 0.752678 |

Text moved more aggressively than images: `true`. Lower cosine means more movement.

## Caption-length slices

Whitespace token groups are fixed at short 1–7, medium 8–12, and long 13+.

| Slice representation | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank | Queries | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short zero-shot | 0.6842 | 0.8772 | 0.9298 | 0.7672 | 1.00 | 3.16 | 57 | 100 |
| Short adapted | 0.6842 | 0.8421 | 0.9474 | 0.7572 | 1.00 | 2.91 | 57 | 100 |
| Medium zero-shot | 0.8313 | 0.9794 | 1.0000 | 0.8922 | 1.00 | 1.42 | 243 | 100 |
| Medium adapted | 0.7778 | 0.9547 | 0.9918 | 0.8555 | 1.00 | 1.64 | 243 | 100 |
| Long zero-shot | 0.8200 | 0.9700 | 0.9750 | 0.8906 | 1.00 | 1.63 | 200 | 100 |
| Long adapted | 0.8100 | 0.9650 | 0.9900 | 0.8793 | 1.00 | 1.52 | 200 | 100 |

## Recorded training behavior

- Selected / stopping epoch: 6 / 10
- Best / final validation mean MRR: 0.898587 / 0.891631
- Initial / selected / final training loss: 2.053616 / 0.638473 / 0.431466
- Loss trend: decreased throughout the recorded run
- Before selection: validation mean MRR ranged from 0.872376 to 0.893435 before epoch 6
- After selection: all 4 later validation values were below the selected value
- Supported interpretation: evidence consistent with overfitting after the selected epoch

## Bounded qualitative examples

### Five largest text-to-image regressions

- `validation-000247-caption-003`: rank 3 → 14 (change +11, margin -0.131832); caption: A woman walks and a little boy walks to the side of her .
- `validation-000848-caption-001`: rank 5 → 15 (change +10, margin -0.172370); caption: A child in a blue shirt jumping off a bench .
- `validation-000414-caption-002`: rank 2 → 11 (change +9, margin -0.137054); caption: Three women are looking into a camera .
- `validation-000414-caption-001`: rank 1 → 9 (change +8, margin -0.066082); caption: Three people are looking into photographic equipment .
- `validation-000848-caption-002`: rank 1 → 8 (change +7, margin -0.091739); caption: A child jumping off bleachers with a blue shirt .

### Five largest text-to-image improvements

- `validation-000928-caption-001`: rank 41 → 2 (change -39, margin +0.050592); caption: A baseball player is on the field in fronmt of an audience .
- `validation-000483-caption-002`: rank 35 → 20 (change -15, margin -0.121700); caption: a man holds a cigarette .
- `validation-000033-caption-001`: rank 10 → 2 (change -8, margin +0.026666); caption: a black and brown dog staring off into the distance at something
- `validation-000049-caption-001`: rank 10 → 3 (change -7, margin -0.015767); caption: A man and woman walking down the street .
- `validation-000426-caption-001`: rank 11 → 4 (change -7, margin -0.054035); caption: a black dog jumping into some water to catch a red and blue Frisbee

### Five largest image-to-text regressions

- `validation-000581`: rank 3 → 6 (change +3, margin -0.033192); caption: A bicyclist is attempting a trick down a flight of outdoor stairs .
- `validation-000842`: rank 1 → 3 (change +2, margin -0.065739); caption: A hockey player in red untangles from a player in white as he goes for the puck .
- `validation-000809`: rank 1 → 3 (change +2, margin -0.048731); caption: A man jumps off a large building onto the ground .
- `validation-000895`: rank 1 → 2 (change +1, margin -0.044902); caption: There is a little blond hair girl with a green sweatshirt and a red shirt playing on a playground .
- `validation-000507`: rank 1 → 2 (change +1, margin -0.040140); caption: A Jack Russell Terrier jumps into a stream .

### Five largest image-to-text improvements

- `validation-000018`: rank 6 → 1 (change -5, margin +0.048149); caption: The biker is riding down a grassy mountainside .
- `validation-000038`: rank 6 → 1 (change -5, margin +0.034873); caption: Two little boys sit in a toy car on the grass .
- `validation-000334`: rank 6 → 2 (change -4, margin +0.010039); caption: A large black and white dog runs in the ocean water
- `validation-000418`: rank 5 → 2 (change -3, margin -0.010777); caption: A child climbs to the top of a slide .
- `validation-000586`: rank 3 → 1 (change -2, margin +0.019201); caption: A boy walks across a rope structure on a playground .

## Supported diagnosis

- **likely overfitting** — training loss fell from 0.638473 at selected epoch 6 to 0.431466 at epoch 10, while all later validation values stayed below the selected value.
- **likely optimization imbalance between modalities** — text-to-image and image-to-text MRR changes differed by 0.023127 (-0.023460 versus -0.000333).
- **representation movement too aggressive** — mean original-to-adapted cosine was 0.685808 for images and 0.644330 for text; at least one fell below 0.95.
- **insufficient improvement signal** — mean bidirectional MRR changed by -0.011897, below the +0.005 promotion requirement.

## Decision and limitations

The Milestone 9A decision remains **retain zero-shot CLIP**. The adapter is not
evaluated on the official test split and is not applied to serving indexes.
This analysis is descriptive and validation-only. It uses one saved configuration
and cannot establish causality or support a new quality claim. No CLIP inference,
embedding generation, retraining, reranking, or parameter search was performed.
