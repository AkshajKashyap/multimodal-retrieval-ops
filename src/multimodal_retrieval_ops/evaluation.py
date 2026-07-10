"""Retrieval evaluation protocol and ranking metrics."""

from dataclasses import dataclass
import statistics

from .baseline_index import IndexEntry, exact_search

EVALUATION_SPLITS = frozenset({"validation", "test"})


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    median_rank: float
    mean_rank: float
    query_count: int


def metrics_from_ranks(ranks: list[int]) -> RetrievalMetrics:
    """Calculate standard single-relevant-target retrieval metrics."""
    if not ranks:
        raise ValueError("at least one query rank is required")
    count = len(ranks)
    return RetrievalMetrics(
        recall_at_1=sum(rank <= 1 for rank in ranks) / count,
        recall_at_5=sum(rank <= 5 for rank in ranks) / count,
        recall_at_10=sum(rank <= 10 for rank in ranks) / count,
        mrr=sum(1.0 / rank for rank in ranks) / count,
        median_rank=float(statistics.median(ranks)),
        mean_rank=statistics.mean(ranks),
        query_count=count,
    )


def evaluate_index(
    vocabulary: list[str],
    entries: list[IndexEntry],
    evaluation_splits: set[str] | frozenset[str] = EVALUATION_SPLITS,
) -> tuple[RetrievalMetrics, list[int]]:
    """Evaluate held-out captions against held-out candidates only."""
    queries = [entry for entry in entries if entry.split in evaluation_splits]
    candidate_count = len(queries)
    if candidate_count == 0:
        raise ValueError("index has no validation/test queries to evaluate")
    ranks: list[int] = []
    for query in queries:
        results = exact_search(
            query.caption,
            vocabulary,
            entries,
            k=candidate_count,
            allowed_splits=set(evaluation_splits),
        )
        ranks.append(next(index for index, result in enumerate(results, start=1) if result.item_id == query.item_id))
    return metrics_from_ranks(ranks), ranks
