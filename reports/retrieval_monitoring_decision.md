# Retrieval Monitoring Decision

Decision: **warning**

- maximum_error_rate: passed=True
- maximum_readiness_failures: passed=True
- minimum_labeled_recall_at_10: passed=False
- minimum_labeled_mrr: passed=True

This is a bounded single-window service-health decision. Known-label thresholds use
cached-ID queries only; arbitrary queries have no inferred relevance labels.
