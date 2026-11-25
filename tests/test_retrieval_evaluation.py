import json
from collections import defaultdict
from pathlib import Path
import unittest

from services.answer_engine import get_top_faq_candidates
from services.retrieval_evaluation import evaluate_retrieval_cases


BACKUP_PATH = (
    Path(__file__).resolve().parents[1]
    / "backups"
    / "faq_backup_20260424_112552.json"
)


def _backup_faq_rows():
    with BACKUP_PATH.open(encoding="utf-8") as file:
        rows = json.load(file)["faqs"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["question"]].append(row["id"])
    faq_rows = [
        (
            row.get("answer"),
            row.get("question"),
            row.get("id"),
            row.get("topic"),
            row.get("intent"),
            row.get("keywords"),
            row.get("related_questions"),
            row.get("priority_score"),
            row.get("image_path"),
            row.get("embedding"),
        )
        for row in rows
    ]
    return faq_rows, grouped


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_report_recall_and_reciprocal_rank(self):
        cases = [
            {"question": "first", "expected_faq_ids": [1], "language": "english"},
            {"question": "second", "expected_faq_ids": [2, 3], "language": "bisaya"},
            {"question": "missing", "expected_faq_ids": [4], "language": "tagalog"},
        ]
        rankings = {
            "first": [1, 2],
            "second": [9, 3],
            "missing": [8, 7],
        }

        report = evaluate_retrieval_cases(
            cases, lambda case: rankings[case["question"]]
        )

        self.assertEqual(report["case_count"], 3)
        self.assertEqual(report["recall_at_1"], 1 / 3)
        self.assertEqual(report["recall_at_3"], 2 / 3)
        self.assertAlmostEqual(report["mrr"], (1 + 0.5 + 0) / 3)
        self.assertEqual(report["by_language"]["bisaya"]["recall_at_3"], 1.0)

    def test_canonical_backup_questions_retrieve_their_faqs(self):
        rows, grouped = _backup_faq_rows()
        cases = [
            {
                "question": question,
                "expected_faq_ids": faq_ids,
                "language": "english",
            }
            for question, faq_ids in grouped.items()
        ]

        report = evaluate_retrieval_cases(
            cases,
            lambda case: [
                candidate["faq"][2]
                for candidate in get_top_faq_candidates(
                    case["question"],
                    case["question"],
                    rows,
                    detected_lang=case["language"],
                )
            ],
        )

        self.assertEqual(report["case_count"], len(grouped))
        self.assertEqual(report["recall_at_3"], 1.0)
        self.assertGreater(report["recall_at_1"], 0.9)


if __name__ == "__main__":
    unittest.main()
