# CLIP Flickr8k Bidirectional Retrieval Report

Run state: **success**

- Dataset source: `jxie/flickr8k`
- Resolved dataset revision: `56f58c967835f7c508d684f36bd7897cca9d7634`
- Dataset fingerprint: `f0e3df07c059a4d5516a2f7e9549240844724656ac3217429f65a44f747327ec`
- Dataset revision: `default`
- Manifest fingerprint: `fc661acaff9522129b263fe95b50d29e12558035028cc4d6d042a0cf2a45c304`
- Official split: `test`
- Benchmark mode: `integration_subset`
- Model: `openai/clip-vit-base-patch32`
- Model revision: `default`
- Device: `cpu`
- Embedding dimension: 512
- Batch size: 16
- Unique images: 100
- Captions: 500
- Cache: hit

| Direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank | Queries | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text to image | 0.8060 | 0.9780 | 0.9920 | 0.8811 | 1.00 | 1.45 | 500 | 100 |
| Image to text | 0.9300 | 1.0000 | 1.0000 | 0.9583 | 1.00 | 1.12 | 100 | 500 |

Image-to-text rank uses the highest-ranked caption relevant to each image.
The dataset source does not clearly expose licensing information; status is unresolved.
