# CLIP Flickr8k Bidirectional Retrieval Report

Run state: **success**

- Dataset source: `jxie/flickr8k`
- Resolved dataset revision: `56f58c967835f7c508d684f36bd7897cca9d7634`
- Dataset fingerprint: `f0e3df07c059a4d5516a2f7e9549240844724656ac3217429f65a44f747327ec`
- Dataset revision: `default`
- Manifest fingerprint: `a254e77e91d14559cade632b78329ab4991971feeff5dc218da1917e15bd3d26`
- Official split: `test`
- Benchmark mode: `complete_test_split`
- Model: `openai/clip-vit-base-patch32`
- Model revision: `default`
- Device: `cpu`
- Embedding dimension: 512
- Batch size: 16
- Unique images: 1000
- Captions: 5000
- Cache: hit

| Direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank | Queries | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text to image | 0.5538 | 0.8160 | 0.8910 | 0.6712 | 1.00 | 5.19 | 5000 | 1000 |
| Image to text | 0.7170 | 0.9170 | 0.9560 | 0.8031 | 1.00 | 2.50 | 1000 | 5000 |

Image-to-text rank uses the highest-ranked caption relevant to each image.
The dataset source does not clearly expose licensing information; status is unresolved.
