import unittest
from unittest.mock import patch

from services import ai_retrieval, faq, handbook, language, nlg, topic_aliases
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

    def test_casual_bisaya_and_tagalog_phrasing_is_detected_and_normalized(self):
        bisaya_questions = (
            "pwede ba mag sout og earings diri sa school",
            "unsaon pagsul-ob ug hikaw dinhi",
            "ngano man dili pwede magsout og hikaw",
            "naa bay klase ugma",
            "tagpila ang tuition sa nursing",
        )
        for question in bisaya_questions:
            with self.subTest(question=question):
                self.assertEqual(_heuristic_language(question)[0], "bisaya")
                if "hikaw" in question or "earings" in question:
                    self.assertIn("earrings", normalize_vernacular_spelling(question))

        tagalog_questions = (
            "pwede ba akong magsuot ng hikaw sa paaralan",
            "bakit hindi puwedeng magsuot ng earrings dito",
            "maaari ba akong mag suot ng earing sa school",
            "pwedeng bumisita muna sa registrar",
        )
        for question in tagalog_questions:
            with self.subTest(question=question):
                self.assertEqual(_heuristic_language(question)[0], "tagalog")
                if "hikaw" in question or "earring" in question:
                    self.assertIn("earring", normalize_vernacular_spelling(question))

    def test_tuition_language_and_semester_phrases_expand_to_search_terms(self):
        tagalog = "magkano tuition sa BSCS 2nd year 1 sem"
        bisaya = "tagpila ang tuition sa BSCS 2nd year 1 sem"

        self.assertEqual(_heuristic_language(tagalog)[0], "tagalog")
        self.assertEqual(_heuristic_language(bisaya)[0], "bisaya")

        expanded = expand_query(normalize_vernacular_spelling(tagalog))
        self.assertIn("how much tuition fee cost", expanded)
        self.assertIn("first semester", expanded)
        self.assertIn("year level", expanded)
        self.assertIn("tuition", normalize_vernacular_spelling("tagpila ang tuision"))

    def test_earring_questions_are_recognized_as_handbook_questions(self):
        for question in (
            "pwede ba mag sout og earings diri sa school",
            "pwede ba akong magsuot ng hikaw sa paaralan",
        ):
            with self.subTest(question=question):
                self.assertTrue(handbook.is_handbook_question(question))

    def test_casual_dress_questions_route_to_the_handbook(self):
        questions = (
            "unsa dapat souton sulod sa school",
            "unsay dress code policy diri sa school",
            "pwede ra ba mag short diri sa school",
            "is it okay to wear short inside the campus",
            "What is the dress code policy?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(handbook.is_handbook_question(question))
                self.assertIn(
                    "dress",
                    handbook.expand_query_for_retrieval(question).lower(),
                )

        with patch.object(topic_aliases, "forced_topic_for", return_value=None):
            for question in questions:
                with self.subTest(forced_topic=question):
                    forced = handbook._forced_topic_for(question)
                    self.assertIsNotNone(forced)
                    self.assertIn("dress and appearance", forced[0])

    def test_bare_campus_mention_does_not_force_campus_location_content(self):
        stale_campus_alias = {
            "id": 123,
            "topic_key": "campus_location",
            "match_pattern": r"\bcampus\b",
            "title_needles": ["meetings"],
            "content_needles": ["held within the campus"],
        }
        with patch.object(
            topic_aliases,
            "_load_aliases",
            return_value=[stale_campus_alias],
        ):
            self.assertEqual(
                topic_aliases.match_aliases(
                    "is it okay to wear short inside the campus",
                    get_db_fn=lambda: None,
                ),
                [],
            )
            self.assertEqual(
                topic_aliases.match_aliases(
                    "where is the main campus",
                    get_db_fn=lambda: None,
                )[0]["topic_key"],
                "campus_location",
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

    def test_bisaya_output_with_shared_magsuot_word_passes_language_check(self):
        response = "Pwede ba magsuot og shorts sulod sa campus?"

        self.assertEqual(_heuristic_language(response)[0], "bisaya")
        self.assertTrue(language.verify_output_language(response, "bisaya"))
        self.assertEqual(
            _heuristic_language("Pwede ba akong magsuot ng hikaw sa paaralan?")[0],
            "tagalog",
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
    def test_tagalog_program_year_semester_tuition_query_reaches_faq_answer(self):
        question = "magkano tuition sa BSCS 2nd year 1 sem"
        faq_row = (
            "Tuition fees vary by program and year level.",
            "How much is the tuition fee?",
            2, "Fees", "tuition", "tuition fee", None, 0, None, None,
        )

        with patch.object(faq, "_get_all_faqs", return_value=[faq_row]), \
             patch.object(faq, "get_embedding", return_value=None), \
             patch.object(
                 faq,
                 "llm_expand_faq_query",
                 return_value=[
                     "How much is tuition for BSCS 2nd year, semester 1?"
                 ],
             ), \
             patch.object(
                 faq.engine,
                 "get_top_faq_candidates",
                 wraps=faq.engine.get_top_faq_candidates,
             ) as rank_candidates, \
             patch.object(
                 faq.engine,
                 "smart_extract_answer",
                 return_value=("Ang tuition ay depende sa programa at year level.", None),
             ) as extract:
            result = faq._lookup_faq_answer(question, "tagalog")

        self.assertEqual(
            result,
            ("Ang tuition ay depende sa programa at year level.", "Fees", "tuition"),
        )
        extract.assert_called_once()
        self.assertGreaterEqual(
            extract.call_args.kwargs["candidates"][0]["combined"],
            faq.engine.FUZZY_MIN_SCORE,
        )
        searched_queries = rank_candidates.call_args.kwargs["alternate_questions"]
        self.assertTrue(
            any("bscs" in query and "semester 1" in query for query in searched_queries)
        )

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

    def test_query_expansion_cannot_drop_a_program_code(self):
        with patch.object(ai_retrieval.ai_cache, "get", return_value=None), \
             patch.object(ai_retrieval.ai_cache, "set") as cache_set, \
             patch.object(
                 ai_retrieval, "llm_chat",
                 return_value='["How much is tuition for second-year students?"]',
             ):
            variants = ai_retrieval.llm_expand_retrieval_query(
                "magkano tuition sa BSCS 2nd year",
                "tagalog",
                "FAQ",
            )

        self.assertEqual(variants, [])
        self.assertEqual(cache_set.call_args.args[1], [])

    def test_query_expansion_cannot_drop_a_negated_policy_intent(self):
        with patch.object(ai_retrieval.ai_cache, "get", return_value=None), \
             patch.object(ai_retrieval.ai_cache, "set") as cache_set, \
             patch.object(
                 ai_retrieval, "llm_chat",
                 return_value='["Can students wear earrings at school?"]',
             ):
            variants = ai_retrieval.llm_expand_retrieval_query(
                "dili ba pwede magsuot og earrings sa school",
                "bisaya",
                "student handbook",
            )

        self.assertEqual(variants, [])
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

    def test_faq_query_expansion_also_runs_when_lexical_match_is_strong(self):
        faq_row = (
            "Tuition varies by program and year level.",
            "How much is the tuition fee?",
            2, "Fees", "tuition", "tuition fee", None, 0, None, None,
        )
        with patch.object(
            faq, "llm_expand_faq_query",
            return_value=["What is the tuition fee for BSCS?"],
        ) as expand:
            faq._get_faq_candidates(
                "How much is the tuition fee?",
                "bisaya",
                [faq_row],
            )

        expand.assert_called_once()

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

    def test_handbook_retrieval_searches_original_and_expansion_together(self):
        chunks = [{"section_title": "Student conduct", "_raw_score": 0.32}]
        with patch.object(
            handbook, "_get_top_handbook_chunks",
            return_value=chunks,
        ) as retrieve, patch.object(
            handbook,
            "llm_expand_school_query",
            return_value=["Student dress and appearance policy"],
        ) as expand:
            result = handbook.get_top_handbook_chunks(
                "unsa pwede ibutang sa dalunggan",
                detected_lang="bisaya",
            )

        self.assertEqual(result, chunks)
        expand.assert_called_once()
        retrieve.assert_called_once()
        self.assertEqual(
            retrieve.call_args.kwargs["alternate_questions"],
            ["Student dress and appearance policy"],
        )

    def test_handbook_retrieval_falls_back_to_original_when_expansion_is_unavailable(self):
        with patch.object(
            handbook, "_get_top_handbook_chunks",
            return_value=[],
        ), patch.object(
            handbook,
            "llm_expand_school_query",
            return_value=[],
        ):
            result = handbook.get_top_handbook_chunks(
                "unsa ang dapat isul-ob sa campus",
                detected_lang="bisaya",
            )

        self.assertEqual(result, [])

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

    def test_bisaya_faq_translation_uses_semantic_source_check(self):
        translated = "Kinahanglan nga simple ug limpyo ang sinina."
        with patch(
            "services.answer_engine.answer_is_grounded", return_value=False
        ) as lexical_check, patch(
            "services.answer_engine._answer_relevant_to_question",
            return_value=False,
        ), patch(
            "services.answer_engine.llm_self_check", return_value=True
        ):
            answer = _extract_from_answer(
                "unsa dapat souton sulod sa school",
                "What is the dress code policy?",
                "Students should dress modestly and wear simple but neat clothing.",
                lambda prompt, max_tokens: translated,
                detected_lang="bisaya",
            )

        self.assertEqual(answer, translated)
        lexical_check.assert_not_called()

    def test_bisaya_handbook_translation_uses_semantic_source_check(self):
        translated = "Kinahanglan nga simple ug limpyo ang sinina."
        chunks = [{
            "section_title": "Dress and appearance",
            "page_start": 25,
            "content": (
                "Students should dress modestly and wear simple but neat clothing."
            ),
        }]
        with patch.object(handbook, "get_relevant_chunks", return_value=chunks), \
             patch.object(handbook.circuit_breaker, "is_open", return_value=False), \
             patch.object(handbook, "llm_chat", return_value=translated), \
             patch.object(handbook, "answer_is_grounded", return_value=False) as lexical_check, \
             patch.object(handbook, "llm_self_check", return_value=True), \
             patch.object(handbook, "_forced_topic_for", return_value=None):
            answer = handbook._build_handbook_answer_from_chunks(
                "unsa dapat souton sulod sa school",
                chunks,
                detected_lang="bisaya",
            )

        self.assertEqual(answer, translated)
        lexical_check.assert_not_called()

    def test_bisaya_natural_rephrase_uses_semantic_source_check(self):
        canonical = (
            "Students should dress modestly and wear simple but neat clothing."
        )
        translated = "Kinahanglan nga simple ug limpyo ang sinina."
        with patch.object(nlg.circuit_breaker, "is_open", return_value=False), \
             patch.object(nlg.ai_cache, "get", return_value=None), \
             patch.object(nlg, "llm_chat_fast", return_value=translated), \
             patch.object(nlg, "answer_is_grounded", return_value=False) as lexical_check, \
             patch("services.ai_retrieval.llm_self_check", return_value=True):
            response = nlg.speak_naturally(
                canonical,
                "unsa dapat souton sulod sa school",
                detected_lang="bisaya",
            )

        self.assertEqual(response, translated)
        lexical_check.assert_not_called()

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