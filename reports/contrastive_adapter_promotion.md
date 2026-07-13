# Contrastive Adapter Promotion Decision

Run state: **success**

Decision: **retain zero-shot CLIP**

Mean bidirectional MRR difference: `-0.011897`

## Conservative gates

- PASS: evaluation used only the untouched official validation subset
- FAIL: mean bidirectional MRR improved by at least 0.005
- PASS: text-to-image Recall@10 did not decrease by more than 0.005
- PASS: image-to-text Recall@10 did not decrease by more than 0.005
- FAIL: text-to-image MRR did not decrease by more than 0.005
- PASS: image-to-text MRR did not decrease by more than 0.005

Recommendation: Retain zero-shot CLIP; do not promote these adapters.

The decision uses only the official validation subset. The official test split remains
untouched and is reserved for a later milestone.
