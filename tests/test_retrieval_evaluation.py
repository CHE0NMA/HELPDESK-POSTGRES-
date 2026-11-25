import unittest

from services.answer_engine import get_top_faq_candidates
from services.retrieval_evaluation import evaluate_retrieval_cases


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

    def test_known_faq_questions_retrieve_their_faqs(self):
        rows = [
            (
                "Submit the request to the registrar.",
                "How do I request a transcript?",
                34, "Registrar", "transcript", "transcript",
                "Unsaon pagkuha sa transcript?", 10, None, None,
            ),
            (
                "The registrar's office is open on weekdays.",
                "What are the registrar's office hours?",
                35, "Registrar", "office_hours", "registrar hours", None, 10, None, None,
            ),
            (
                "Tuition fees depend on the program and year level.",
                "How much is the tuition fee?",
                36, "Fees", "tuition", "tuition fee", None, 10, None, None,
            ),
            (
                "The academic calendar is available from the registrar.",
                "Where can I find the academic calendar?",
                37, "Registrar", "calendar", "academic calendar", None, 10, None, None,
            ),
            (
                "Use the password reset link on the student portal sign-in page.",
                "How do I reset my student portal password?",
                38, "Student Portal", "password_reset", "portal password", None, 10, None, None,
            ),
        ]
        cases = [
            {
                "question": row[1],
                "expected_faq_ids": [row[2]],
                "language": "english",
            }
            for row in rows
        ]
        cases.append({
            "question": "Unsaon pagkuha sa transcript?",
            "expected_faq_ids": [34],
            "language": "bisaya",
        })

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

        self.assertEqual(report["case_count"], len(cases))
        self.assertEqual(report["recall_at_3"], 1.0)
        self.assertGreater(report["recall_at_1"], 0.9)


if __name__ == "__main__":
    unittest.main()
