"""Metrics for measuring FAQ retrieval against reviewed FAQ IDs."""
from collections import defaultdict
from typing import Callable, Dict, Iterable, Sequence


def evaluate_retrieval_cases(
    cases: Iterable[Dict],
    rank_case: Callable[[Dict], Sequence[int]],
    k_values: Sequence[int] = (1, 3),
) -> Dict:
    """Compute Recall@k and MRR for cases with one or more acceptable FAQ IDs."""
    normalized_k = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not normalized_k:
        raise ValueError("k_values must contain at least one positive integer")

    totals = defaultdict(lambda: {"count": 0, "hits": defaultdict(int), "rr": 0.0})
    for case in cases:
        question = str(case.get("question") or "").strip()
        expected = case.get("expected_faq_ids", case.get("expected_faq_id"))
        if isinstance(expected, (str, int)):
            expected = [expected]
        expected_ids = {int(value) for value in (expected or [])}
        if not question or not expected_ids:
            raise ValueError("Each case needs a question and at least one expected FAQ ID")

        language = str(case.get("language") or "unknown").lower()
        ranked_ids = [int(value) for value in rank_case(case)]
        first_relevant_rank = next(
            (rank for rank, faq_id in enumerate(ranked_ids, start=1)
             if faq_id in expected_ids),
            None,
        )
        bucket = totals[language]
        bucket["count"] += 1
        if first_relevant_rank is not None:
            bucket["rr"] += 1.0 / first_relevant_rank
        for k in normalized_k:
            if any(faq_id in expected_ids for faq_id in ranked_ids[:k]):
                bucket["hits"][k] += 1

    def summarize(bucket):
        count = bucket["count"]
        return {
            "case_count": count,
            **{
                f"recall_at_{k}": bucket["hits"][k] / count if count else 0.0
                for k in normalized_k
            },
            "mrr": bucket["rr"] / count if count else 0.0,
        }

    combined = defaultdict(lambda: {"count": 0, "hits": defaultdict(int), "rr": 0.0})
    for bucket in totals.values():
        combined["all"]["count"] += bucket["count"]
        combined["all"]["rr"] += bucket["rr"]
        for k in normalized_k:
            combined["all"]["hits"][k] += bucket["hits"][k]

    overall = summarize(combined["all"])
    overall["by_language"] = {
        language: summarize(bucket)
        for language, bucket in sorted(totals.items())
    }
    return overall
