# Model Card

## Model and intended use

The promoted neural path uses `openai/clip-vit-base-patch32` without fine-tuning. It is intended as
a local reference for bidirectional image-text retrieval evaluation and persisted-index serving,
not as a safety-critical classifier, biometric system, content moderation authority, or claim of
production-scale deployment.

## Evaluation

The main benchmark is the official Flickr8k test split: 1,000 images and 5,000 captions. Every
caption queries one image candidate per image; every image queries all caption candidates and is
correct when any associated caption is retrieved. Tracked zero-shot results are:

| Direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text to image | 0.5538 | 0.8160 | 0.8910 | 0.6712 | 1.00 | 5.19 |
| Image to text | 0.7170 | 0.9170 | 0.9560 | 0.8031 | 1.00 | 2.50 |

These are local tracked results, not universal quality claims. FlatIP reproduced the reference
rankings. A bounded validation-only residual-adapter experiment changed mean bidirectional MRR by
`-0.011897`; it failed conservative promotion gates and was rejected. Zero-shot CLIP remains the
release recommendation.

## Limitations and responsible use

CLIP inherits biases from its pretraining data and can associate images and language incorrectly,
especially across cultures, languages, uncommon concepts, text in images, fine-grained identity,
and distribution shifts. Flickr8k is small and dated, and its tracked upstream licensing status is
unresolved. Results do not establish fairness, robustness, safety, or performance at larger scale.

The full official cache contains 512-dimensional vectors for 1,000 images and 5,000 captions.
Encoding on CPU is substantially more expensive than cached search; memory and latency vary by
hardware and are not release promises. Neural execution requires separately installed dependencies
and locally available weights when downloads are disabled.

Uploads are decoded in memory under byte, pixel, and format bounds and are not persisted by the
service. Telemetry does not retain raw query text or image content; it stores bounded metadata and
safe hashes. Operators remain responsible for access control, retention, encryption, consent, and
threat modeling. The project is unsupported for surveillance, identity inference, autonomous high-
impact decisions, or exposed public operation without substantial security and reliability work.
