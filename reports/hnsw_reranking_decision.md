# HNSW Exact Reranking Decision

Decision: **do not promote**

- Flat top-10 shortlist coverage >= 0.99 in both directions: False
- Reranked top-10 agreement improved in both directions: False
- Reranked MRR preserved within 0.001: True
- Reranked Recall@10 preserved within 0.001: True
- Artifact compatibility passed: True
- Rejected adapter embeddings used: False

Keep the existing raw FlatIP and HNSW serving behavior.

This decision applies only to possible later service work. Milestone 10A does not
modify the retrieval service.
