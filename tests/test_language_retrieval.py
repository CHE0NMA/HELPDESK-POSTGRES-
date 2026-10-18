import unittest
from unittest.mock import patch

from services import faq, handbook, language, nlg
from services.answer_engine import build_disambiguation, get_top_faq_candidates
from services.language import (
    _heuristic_language,
    expand_query,
    normalize_vernacular_spelling,
)


class BisayaLanguageTests(unittest.TestCase):
    def test_casual_bisaya_tuition_questions_are_not_classified_as_english(self):
        for question in (
            "pila ang tuition sa nursing",
            "pila ang tuition gikan sa 1st year hangtud sa 4th year",
            "pila ang tuition sa BSIT 4th year",
        ):
            with self.subTest(question=question):
                self.assertEqual(_heuristic_language(question)[0], "bisaya")

    def test_bisaya_dress_and_earrings_questions_are_not_classified_as_tagalog(self):
        self.assertEqual(
            _heuristic_language("unsay dapat suoton sulod sa school bai")[0],
            "bisaya",
        )

    def test_normalized_bisaya_dress_query_matches_uniform_source(self):
        self.assertTrue(
            handbook._answer_relevant_to_question(
                "unsay dapat suoton sulod sa school bai",
                "Students should choose simple and appropriate attire.",
            )
        )

    def test_tagalog_question_is_still_classified_as_tagalog(self):
        self.assertEqual(
            _heuristic_language("pwede ba akong magsuot ng earrings?")[0],
            "tagalog",
        )
        self.assertEqual(
            _heuristic_language(
                "pwede bako mag sout og earrings sulod sa school?"
            )[0],
            "bisaya",
        )

    def test_bisaya_search_phrases_expand_to_matching_concepts(self):
        tuition = expand_query(normalize_vernacular_spelling(
            "pila ang tuition sa nursing"
        ))
        year_range = expand_query(normalize_vernacular_spelling(
            "gikan sa 1st year hangtud sa 4th year"
        ))

        self.assertIn("how much", tuition)
        self.assertIn("from", year_range)
        self.assertIn("until", year_range)


class FaqRelevanceTests(unittest.TestCase):
    def test_tuition_question_outranks_installment_faq_priority(self):
        faqs = [
            (
                "Installment plans are available.",
                "Can I pay tuition by installment?",
                1, "Fees", "payment", "tuition", None, 91, None, None,
            ),
            (
                "Tuition varies by program and year level.",
                "How much is the tuition fee?",
                2, "Fees", "tuition", "tuition fee", None, 93, None, None,
            ),
        ]
        question = "pila ang tuition sa nursing"
        search_question = normalize_vernacular_spelling(question)
        candidates = get_top_faq_candidates(
            search_question,
            expand_query(search_question),
            faqs,
            detected_lang="bisaya",
        )

        self.assertEqual(candidates[0]["faq"][1], "How much is the tuition fee?")

    def test_mixed_language_disambiguation_is_not_english_only(self):
        candidates = [
            {"combined": 0.65, "best_q_text": "Tuition fee", "faq": ("", "", 1)},
            {"combined": 0.63, "best_q_text": "Payment schedule", "faq": ("", "", 2)},
        ]

        self.assertIn("Naa koy", build_disambiguation(candidates, "mixed_be"))
        self.assertIn("Naa koy", build_disambiguation(candidates, "mixed_tb"))


class NonEnglishFallbackTests(unittest.TestCase):
    def test_bisaya_rephrase_failure_does_not_return_english_source(self):
        english_source = "Tuition varies by program and year level."
        with patch.object(nlg.circuit_breaker, "is_open", return_value=True):
            response = nlg.speak_naturally(
                english_source,
                "pila ang tuition sa nursing",
                detected_lang="bisaya",
            )

        self.assertEqual(
            response,
            language.get_translation_unavailable_response("bisaya"),
        )
        self.assertNotIn("Tuition varies", response)

    def test_tagalog_rephrase_failure_does_not_return_english_source(self):
        english_source = "Tuition varies by program and year level."
        with patch.object(nlg.circuit_breaker, "is_open", return_value=True):
            response = nlg.speak_naturally(
                english_source,
                "magkano ang tuition sa nursing",
                detected_lang="tagalog",
            )

        self.assertEqual(
            response,
            language.get_translation_unavailable_response("tagalog"),
        )
        self.assertNotIn("Tuition varies", response)

    def test_low_confidence_faq_does_not_return_english_source_in_bisaya(self):
        faq_row = (
            "Installment plans are available.",
            "Can I pay tuition by installment?",
            1, "Fees", "payment", "tuition", None, 91, None, None,
        )
        candidate = {"combined": 0.59, "fuzzy": 0.61, "faq": faq_row}
        with patch.object(faq, "_get_all_faqs", return_value=[faq_row]), \
             patch.object(faq.engine, "get_top_faq_candidates", return_value=[candidate]), \
             patch.object(faq.circuit_breaker, "is_open", return_value=True):
            result = faq._lookup_faq_answer("pila ang tuition", "bisaya")

        self.assertEqual(
            result[0],
            language.get_translation_unavailable_response("bisaya"),
        )

    def test_followup_provider_failure_is_localized(self):
        with patch.object(faq.circuit_breaker, "is_open", return_value=True):
            response = faq.answer_followup_with_ai(
                "pila pud sa second year?",
                detected_lang="bisaya",
            )

        self.assertEqual(
            response,
            language.get_translation_unavailable_response("bisaya"),
        )

    def test_missing_bisaya_admin_response_uses_bisaya_default(self):
        class EmptyCursor:
            def execute(self, query, params=None):
                self.query = query

            def fetchone(self):
                return None

            def close(self):
                pass

        class EmptyConnection:
            def __init__(self):
                self.cur = EmptyCursor()

            def cursor(self):
                return self.cur

            def close(self):
                pass

        connection = EmptyConnection()
        with patch.object(language, "get_db", return_value=connection), \
             patch.dict(language._DB_RESPONSE_CACHE, {}, clear=True):
            response = language.get_fallback_response("bisaya")

        self.assertEqual(response, language.LANGUAGE_CONFIG["bisaya"]["fallback"])


if __name__ == "__main__":
    unittest.main()