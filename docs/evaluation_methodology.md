# Evaluation Methodology

## Dataset and relevance

The official Flickr8k train, validation, and test assignments are preserved. Schema v2 models
multiple captions per image with distinct `caption_id` values and a shared `image_id`; image groups
never cross splits. The official test benchmark contains 1,000 images and 5,000 captions.

Text-to-image evaluation issues every caption as a query against one vector per image. The paired
`image_id` is relevant. Image-to-text issues every image as a query against all caption vectors;
any caption attached to that image is relevant, and rank is the first relevant caption.

Recall@K is the fraction of queries with a relevant result in the first K ranks. MRR averages the
reciprocal first-relevant rank. Median and mean rank summarize that first-relevant rank. Reported
query and candidate counts make the denominator explicit.

## Retrieval comparisons

Normalized inner product equals cosine similarity, so exact matrix search and `IndexFlatIP` are the
correctness references. FlatIP had 1.0 top-1, top-5-set, and top-10-set agreement in both directions
with maximum score difference `0.0000002980` from the reference. HNSW is an optional approximation;
the bounded comparison tested `efSearch` 16, 32, and 64, and selected 64 under its two-direction
accuracy gate. Its machine-specific timing is non-authoritative and does not show universal
acceleration at this dataset size.

The adapter configuration was selected and diagnosed only on a bounded official validation subset.
The official test split remained untouched throughout adapter development. Because the adapter
reduced validation mean bidirectional MRR, it was rejected; the existing official zero-shot test
evaluation remains the headline benchmark.

The reranking experiment rescored HNSW candidate vectors exactly. With `IndexHNSWFlat`, raw HNSW
scores already use the stored exact vectors, so rescoring did not improve rankings and the feature
was not promoted.

## Evidence boundaries

- Official test evaluation: the tracked Flickr8k zero-shot CLIP and FlatIP correctness values.
- Bounded validation experiments: adapter training/diagnostics and its rejection decision.
- Synthetic tests: deterministic fixtures proving schemas, cache validation, metrics, and errors.
- Local service smoke: endpoint and artifact compatibility integration, not model quality or SLA.

Telemetry quality metrics exist only for requests with known relevance. Minimum sample safeguards
return `insufficient_data` when evidence is too small; unlabeled traffic, fixture smoke, and local
latencies cannot justify a production-health claim.
