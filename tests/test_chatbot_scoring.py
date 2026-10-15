import unittest

from test_chatbot import classify_answer_outcome


class ChatbotOutcomeTests(unittest.TestCase):
    def test_request_errors_are_separate(self):
        self.assertEqual(
            classify_answer_outcome("[ERROR] connection refused"),
            "request_error",
        )

    def test_polite_give_up_is_a_fallback(self):
        self.assertEqual(
            classify_answer_outcome(
                "I couldn't find a solid answer. I've saved your question "
                "so our staff can add it soon."
            ),
            "fallback",
        )

    def test_missing_expected_fact_is_incorrect(self):
        self.assertEqual(
            classify_answer_outcome(
                "The school history mentions its founding.",
                must_contain=["mariano", "margie"],
            ),
            "incorrect",
        )

    def test_expected_facts_are_explicitly_verified(self):
        self.assertEqual(
            classify_answer_outcome(
                "Mariano Macias and Margie Macias founded the school.",
                must_contain=["mariano", "margie"],
            ),
            "verified",
        )

    def test_answer_without_fact_assertions_is_unverified(self):
        self.assertEqual(
            classify_answer_outcome("A complete answer with no configured facts."),
            "unverified",
        )

    def test_forbidden_content_guardrail_does_not_prove_accuracy(self):
        self.assertEqual(
            classify_answer_outcome(
                "No class suspensions were announced.",
                must_not_contain=["bulletin boards"],
            ),
            "unverified",
        )


if __name__ == "__main__":
    unittest.main()