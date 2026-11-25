import json
import unittest
from unittest.mock import patch

from services import ai_retrieval, answer_engine, faq, handbook


class QuestionUnderstandingTests(unittest.TestCase):
    def test_interpretation_captures_requested_focus_not_topic_rules(self):
        examples = (
            (
                "Who wrote the PAC Hymn?",
                {
                    "meaning": "The student wants to know the credited writer.",
                    "answer_focus": "Name the person credited with writing the hymn.",
                    "constraints": ["Do not provide the lyrics."],
                    "search_queries": ["PAC Hymn writer credit"],
                },
            ),
            (
                "Where can I submit my late enrollment form?",
                {
                    "meaning": "The student wants the submission location.",
                    "answer_focus": "Give the place or office where the form is submitted.",
                    "constraints": ["The form is for late enrollment."],
                    "search_queries": ["late enrollment form submission office"],
                },
            ),
            (
                "Why does the dorm close at 9 PM?",
                {
                    "meaning": "The student asks for the reason behind the dorm closing time.",
                    "answer_focus": "Explain the stated reason for the 9 PM dorm closing time.",
                    "constraints": ["Keep the answer about the reason, not just the time."],
                    "search_queries": ["reason dormitory closes at 9 PM"],
                },
            ),
            (
                "How do I request a transcript?",
                {
                    "meaning": "The student wants the transcript request process.",
                    "answer_focus": "Give the steps to request a transcript.",
                    "constraints": [],
                    "search_queries": ["transcript request procedure"],
                },
            ),
            (
                "What documents do I need to enroll?",
                {
                    "meaning": "The student wants the enrollment document requirements.",
                    "answer_focus": "List the documents required for enrollment.",
                    "constraints": ["Do not give unrelated enrollment steps."],
                    "search_queries": ["required enrollment documents"],
                },
            ),
            (
                "When does registration start?",
                {
                    "meaning": "The student asks for the registration start time or date.",
                    "answer_focus": "Give the registration start date or time.",
                    "constraints": [],
                    "search_queries": ["registration start date"],
                },
            ),
        )

        for question, result in examples:
            with self.subTest(question=question), \
                 patch.object(ai_retrieval.ai_cache, "get", return_value=None), \
                 patch.object(ai_retrieval.ai_cache, "set"), \
                 patch.object(ai_retrieval.circuit_breaker, "is_open", return_value=False), \
                 patch.object(ai_retrieval, "llm_chat", return_value=json.dumps(result)) as chat:
                understanding = ai_retrieval.understand_question(question, "english")

            self.assertIsNotNone(understanding)
            self.assertEqual(understanding.answer_focus, result["answer_focus"])
            self.assertEqual(understanding.search_queries, result["search_queries"])
            self.assertIn(question, chat.call_args.args[0])

    def test_malformed_interpretation_fails_open_to_existing_retrieval(self):
        with patch.object(ai_retrieval.ai_cache, "get", return_value=None), \
             patch.object(ai_retrieval.circuit_breaker, "is_open", return_value=False), \
             patch.object(ai_retrieval, "llm_chat", return_value="not json"):
            self.assertIsNone(
                ai_retrieval.understand_question("Who wrote the hymn?", "english")
            )

    def test_interpretation_search_query_cannot_drop_specific_constraints(self):
        response = {
            "meaning": "The student asks about a restriction.",
            "answer_focus": "Explain whether the requested item is allowed.",
            "constraints": ["BSCS", "second year", "negated request"],
            "search_queries": ["Are students allowed to wear uniforms?"],
        }
        with patch.object(ai_retrieval.ai_cache, "get", return_value=None), \
             patch.object(ai_retrieval.ai_cache, "set"), \
             patch.object(ai_retrieval.circuit_breaker, "is_open", return_value=False), \
             patch.object(ai_retrieval, "llm_chat", return_value=json.dumps(response)):
            understanding = ai_retrieval.understand_question(
                "dili ba pwede sa BSCS 2nd year?", "bisaya"
            )

        self.assertIsNotNone(understanding)
        self.assertEqual(understanding.search_queries, [])

    def test_faq_answer_prompt_and_self_check_receive_answer_focus(self):
        understanding = ai_retrieval.QuestionUnderstanding(
            "The student asks who wrote the hymn.",
            "Name the credited writer, not the lyrics.",
            ["PAC Hymn writer"],
            ["Do not provide lyrics."],
        )
        prompts = []

        def generate(prompt, **kwargs):
            prompts.append(prompt)
            return "L.V. Pilon composed the music."

        with patch.object(answer_engine, "answer_is_grounded", return_value=True), \
             patch.object(answer_engine, "verify_output_language", return_value=True), \
             patch.object(answer_engine, "llm_self_check", return_value=True) as check:
            answer = answer_engine._extract_from_answer(
                "Who wrote the PAC Hymn?",
                "PAC Hymn",
                "Music composed by L.V. Pilon; lyrics written by SL La Sage.",
                generate,
                understanding=understanding,
            )

        self.assertEqual(answer, "L.V. Pilon composed the music.")
        self.assertIn(understanding.answer_focus, prompts[0])
        check.assert_called_once_with(
            "Who wrote the PAC Hymn?",
            answer,
            "Music composed by L.V. Pilon; lyrics written by SL La Sage.",
            understanding,
        )

    def test_interpretation_search_queries_are_added_to_both_retrievers(self):
        faq_row = (
            "Music composed by L.V. Pilon.",
            "Who wrote the PAC Hymn?",
            1,
            "PAC Hymn",
            "school_history",
            "",
            "",
            0,
            None,
            None,
        )
        query = "Kinsay nagsulat sa PAC Hymn?"
        variant = "PAC Hymn writer credit"

        with patch.object(faq.lang_service, "expand_query", side_effect=lambda value: value), \
             patch.object(faq.lang_service, "normalize_vernacular_spelling", side_effect=lambda value: value), \
             patch.object(faq.hybrid_search, "search_faq_candidate_ids", return_value=None), \
             patch.object(faq.engine, "get_top_faq_candidates", return_value=[]) as rank:
            faq._get_faq_candidates(
                query, "bisaya", [faq_row], query_embedding=None,
                query_variants=[variant],
            )

        self.assertIn(variant.lower(), rank.call_args.kwargs["alternate_questions"])

        with patch.object(handbook, "_get_top_handbook_chunks", return_value=[]) as search, \
             patch.object(handbook, "llm_expand_school_query") as expand:
            handbook.get_top_handbook_chunks(
                query, detected_lang="bisaya", query_variants=[variant]
            )

        self.assertEqual(search.call_args.kwargs["alternate_questions"], [variant])
        expand.assert_not_called()

    def test_handbook_uses_interpretation_and_fails_closed_without_ai(self):
        understanding = ai_retrieval.QuestionUnderstanding(
            "The student asks who wrote the hymn.",
            "Name the credited writer, not the lyrics.",
            ["PAC Hymn writer"],
            [],
        )
        chunk = {
            "section_title": "The PAC Hymn",
            "page_start": 7,
            "content": "Music composed by L.V. Pilon; lyrics written by SL La Sage.",
        }
        prompts = []

        with patch.object(handbook, "get_relevant_chunks", return_value=[chunk]), \
             patch.object(handbook.circuit_breaker, "is_open", return_value=False), \
             patch.object(handbook, "llm_chat", side_effect=lambda prompt, **kwargs: (
                 prompts.append(prompt) or "L.V. Pilon composed the music."
             )), \
             patch.object(handbook, "answer_is_grounded", return_value=True), \
             patch.object(handbook, "llm_self_check", return_value=True), \
             patch.object(handbook, "_forced_topic_for", return_value=None):
            answer = handbook._build_handbook_answer_from_chunks(
                "Who wrote the PAC Hymn?",
                [chunk],
                understanding=understanding,
            )

        self.assertEqual(answer, "L.V. Pilon composed the music.")
        self.assertIn(understanding.answer_focus, prompts[0])

        with patch.object(handbook, "get_relevant_chunks", return_value=[chunk]), \
             patch.object(handbook.circuit_breaker, "is_open", return_value=True):
            self.assertIsNone(
                handbook._build_handbook_answer_from_chunks(
                    "Who wrote the PAC Hymn?",
                    [chunk],
                    understanding=understanding,
                )
            )

    def test_self_check_rejects_related_but_wrong_intent(self):
        question = "Who wrote the PAC Hymn?"
        answer = "Here are the lyrics."
        with patch.object(ai_retrieval.circuit_breaker, "is_open", return_value=False), \
             patch.object(ai_retrieval, "llm_chat", return_value="NO") as check:
            self.assertFalse(
                ai_retrieval.llm_self_check(
                    question,
                    answer,
                    "Lyrics written by SL La Sage.",
                )
            )

        prompt = check.call_args.args[0]
        self.assertIn("directly answer the exact question", prompt)
        self.assertIn("substitutes a related fact", prompt)


if __name__ == "__main__":
    unittest.main()
