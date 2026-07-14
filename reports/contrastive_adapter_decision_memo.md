# Contrastive Adapter Decision Memo

Run state: **success**

Decision: **retain zero-shot CLIP**

The validation-only diagnostics reproduced the rejected Milestone 9A result.
Supported findings:

- likely overfitting: training loss fell from 0.638473 at selected epoch 6 to 0.431466 at epoch 10, while all later validation values stayed below the selected value.
- likely optimization imbalance between modalities: text-to-image and image-to-text MRR changes differed by 0.023127 (-0.023460 versus -0.000333).
- representation movement too aggressive: mean original-to-adapted cosine was 0.685808 for images and 0.644330 for text; at least one fell below 0.95.
- insufficient improvement signal: mean bidirectional MRR changed by -0.011897, below the +0.005 promotion requirement.

No second model configuration was trained. The adapter remains excluded from serving
indexes, and the official test split remains untouched for a later milestone.
