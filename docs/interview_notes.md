# Interview Notes

- Problem: make bidirectional image-text retrieval reproducible from dataset relationships through
  evaluation, persisted search, serving, and monitoring.
- Architecture: artifact fingerprints and compatibility checks separate expensive optional encoding
  from lightweight deterministic search and service verification.
- Similarity: L2-normalized vectors make inner product equal cosine similarity.
- FlatIP versus HNSW: FlatIP is exact and the correctness oracle; HNSW trades bounded recall for
  optional approximate search. It is not necessarily faster at Flickr8k scale.
- Reranking: `IndexHNSWFlat` already scores shortlisted stored vectors exactly, so rescoring the same
  candidates did not improve their order.
- Adapter: a single bounded validation-only experiment reduced mean bidirectional MRR by 0.011897;
  conservative gates rejected it and preserved zero-shot CLIP.
- Leakage: official splits remain fixed, captions are grouped by image, vocabulary fitting uses
  train only, and adapter selection never used the official test split.
- Monitoring: threshold decisions need minimum samples; otherwise random small-sample variation can
  produce false confidence. The smoke correctly concludes `insufficient_data`.
- Privacy: telemetry omits raw queries and images, retaining bounded operational metadata and safe
  hashes; uploads are decoded in memory rather than persisted.
- Biggest limits: small benchmark, unresolved upstream dataset license, optional local model weights,
  no production deployment/security envelope, and no evidence at internet scale.
- Next at larger scale: evaluate representative licensed data, establish SLOs and load tests, add
  authentication and threat controls, then compare distributed vector stores or compressed indexes
  under measured quality/cost gates.
