import unittest
from unittest.mock import patch

from services import ai_retrieval, faq, handbook, language, nlg
from services.answer_engine import (
    _extract_from_answer,
    build_disambiguation,
    get_top_faq_candidates,
)
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
    def test_ai_rephrasing_improves_tagalog_and_bisaya_faq_matching(self):
        faq_row = (
            "Students may wear simple earrings.",
            "Are students allowed to wear earrings at school?",
            3, "Student Conduct", "dress_code", "earrings,jewelry",
            None, 90, None, None,
        )
        cases = (
            (
                "unsa pwede ibutang sa dalunggan sulod sa school bai",
                "bisaya",
            ),
            (
                "pwede ba akong magsuot ng hikaw sa campus",
                "tagalog",
            ),
        )

        for question, detected_lang in cases:
            with self.subTest(language=detected_lang), patch.object(
                faq,
                "llm_expand_faq_query",
                return_value=["Are students allowed to wear earrings at school?"],
            ):
                candidates = faq._get_faq_candidates(
                    question, detected_lang, [faq_row]
                )

            self.assertEqual(candidates[0]["faq"], faq_row)
            self.assertTrue(candidates[0]["alternate_match"])
            self.assertGreaterEqual(candidates[0]["combined"], 0.60)

    def test_ai_faq_query_expansion_interprets_intent_before_search(self):
        with patch.object(ai_retrieval.ai_cache, "get", return_value=None), \
             patch.object(ai_retrieval.ai_cache, "set") as cache_set, \
             patch.object(
                 ai_retrieval, "llm_chat",
                 return_value='["How much is the nursing tuition fee?"]',
             ) as llm_chat:
            variants = ai_retrieval.llm_expand_faq_query(
                "pila bayran sa nursing kada tuig", "bisaya"
            )

        self.assertEqual(variants, ["How much is the nursing tuition fee?"])
        prompt = llm_chat.call_args_list[0].args[0]
        self.assertIn("likely intent and school topic", prompt)
        self.assertIn("Bisaya/Cebuano", prompt)
        cache_set.assert_called_once()

    def test_query_expansion_cannot_drop_a_requested_year_level(self):
        with patch.object(ai_retrieval.ai_cache, "get", return_value=None), \
             patch.object(ai_retrieval.ai_cache, "set") as cache_set, \
             patch.object(
                 ai_retrieval, "llm_chat",
                 return_value='["How much is nursing tuition?"]',
             ):
            variants = ai_retrieval.llm_expand_retrieval_query(
                "pila tuition sa nursing 4th year",
                "bisaya",
                "FAQ",
            )

        self.assertEqual(variants, [])
        cache_set.assert_called_once()
        self.assertEqual(cache_set.call_args.args[1], [])

    def test_retrieval_uses_reviewed_student_paraphrase_alias(self):
        faq_row = (
            "Students may wear simple earrings.",
            "Are students allowed to wear earrings at school?",
            3, "Student Conduct", "dress_code", "earrings,jewelry",
            "unsa pwede ibutang sa dalunggan sulod sa school bai",
            90, None, None,
        )
        candidates = faq._get_faq_candidates(
            "unsa pwede ibutang sa dalunggan sulod sa school bai",
            "bisaya",
            [faq_row],
        )

        self.assertEqual(candidates[0]["faq"], faq_row)
        self.assertTrue(candidates[0]["alternate_match"])

    def test_faq_query_expansion_only_runs_when_lexical_match_is_weak(self):
        faq_row = (
            "Tuition varies by program and year level.",
            "How much is the tuition fee?",
            2, "Fees", "tuition", "tuition fee", None, 0, None, None,
        )
        with patch.object(faq, "llm_expand_faq_query") as expand:
            faq._get_faq_candidates(
                "How much is the tuition fee?",
                "bisaya",
                [faq_row],
            )

        expand.assert_not_called()

    def test_faq_query_expansion_skips_when_faq_corpus_is_empty(self):
        with patch.object(faq, "llm_expand_faq_query") as expand:
            result = faq._get_faq_candidates(
                "pila bayran sa nursing",
                "bisaya",
                [],
            )

        self.assertEqual(result, [])
        expand.assert_not_called()

    def test_semantic_blend_does_not_cancel_a_confident_ai_rephrase_match(self):
        faq_row = (
            "Students may wear simple earrings.",
            "Are students allowed to wear earrings at school?",
            3, "Student Conduct", "dress_code", None, None, 0, None, [0.0],
        )
        candidates = get_top_faq_candidates(
            "unsa pwede ibutang sa dalunggan",
            "unsa pwede ibutang sa dalunggan",
            [faq_row],
            detected_lang="bisaya",
            query_embedding=[0.0],
            alternate_questions=[
                "Are students allowed to wear earrings at school?"
            ],
        )

        self.assertGreaterEqual(candidates[0]["combined"], 0.60)
        self.assertTrue(candidates[0]["alternate_match"])

    def test_handbook_query_expansion_recovers_weak_multilingual_hits(self):
        initial_chunks = [{"section_title": "General rules", "_raw_score": 0.08}]
        expanded_chunks = [{"section_title": "Student conduct", "_raw_score": 0.32}]
        with patch.object(
            handbook, "_get_top_handbook_chunks",
            side_effect=[initial_chunks, expanded_chunks],
        ) as retrieve, patch.object(
            handbook,
            "llm_expand_retrieval_query",
            return_value=["Student dress and appearance policy"],
        ) as expand:
            result = handbook.get_top_handbook_chunks(
                "unsa pwede ibutang sa dalunggan",
                detected_lang="bisaya",
            )

        self.assertEqual(result, expanded_chunks)
        expand.assert_called_once()
        self.assertEqual(
            retrieve.call_args.kwargs["alternate_questions"],
            ["Student dress and appearance policy"],
        )

    def test_handbook_query_expansion_skips_strong_hits(self):
        chunks = [{"section_title": "Dress and appearance", "_raw_score": 0.42}]
        with patch.object(
            handbook, "_get_top_handbook_chunks", return_value=chunks
        ), patch.object(handbook, "llm_expand_retrieval_query") as expand:
            result = handbook.get_top_handbook_chunks(
                "unsa ang dress code",
                detected_lang="bisaya",
            )

        self.assertEqual(result, chunks)
        expand.assert_not_called()

    def test_handbook_query_expansion_runs_when_initial_retrieval_has_no_hits(self):
        expanded_chunks = [{"section_title": "Student conduct", "_raw_score": 0.16}]
        with patch.object(
            handbook, "_get_top_handbook_chunks",
            side_effect=[[], expanded_chunks],
        ), patch.object(
            handbook,
            "llm_expand_retrieval_query",
            return_value=["Student dress and appearance policy"],
        ):
            result = handbook.get_top_handbook_chunks(
                "unsa ang dapat isul-ob sa campus",
                detected_lang="bisaya",
            )

        self.assertEqual(result, expanded_chunks)

    def test_unanswered_log_stores_detected_language_and_topic(self):
        class Cursor:
            def execute(self, query, params):
                self.query = query
                self.params = params

            def close(self):
                pass

        class Connection:
            def __init__(self):
                self.cur = Cursor()

            def cursor(self):
                return self.cur

            def commit(self):
                pass

            def close(self):
                pass

        connection = Connection()
        with patch.object(faq, "get_db", return_value=connection):
            faq._log_unanswered(
                "pila ang tuition sa nursing",
                detected_lang="bisaya",
                detected_topic="tuition",
            )

        self.assertIn("detected_language", connection.cur.query)
        self.assertIn("detected_topic", connection.cur.query)
        self.assertEqual(connection.cur.params[-2:], ("bisaya", "tuition"))

    def test_specific_item_is_not_answered_from_a_general_policy_source(self):
        with patch("services.answer_engine.llm_self_check") as self_check:
            answer = _extract_from_answer(
                "pwede ba akong magsuot ng hikaw?",
                "What attire is appropriate?",
                "Students should choose simple and appropriate attire.",
                lambda prompt, max_tokens: "You may wear earrings.",
                detected_lang="english",
            )

        self.assertIsNone(answer)
        self_check.assert_not_called()

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