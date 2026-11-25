import unittest
from unittest.mock import patch

from services import faq, handbook, nlg
from services.conversation import ConversationContext


class QuestionContextTests(unittest.TestCase):
    def test_complete_questions_are_not_followups(self):
        examples = (
            "What's the requirements for enrollment",
            "Is PAC a Christian school",
            "How much is tuition for BSBA?",
        )
        for question in examples:
            with self.subTest(question=question), patch.object(
                nlg, "llm_chat_fast"
            ) as chat:
                self.assertFalse(
                    nlg.classify_followup(
                        question,
                        "Is there school on Saturday?",
                        "Saturday is a closed day.",
                    )
                )
                chat.assert_not_called()

    def test_explicit_followup_phrasings_remain_followups(self):
        for question in ("How about the second year?", "pila ang sa second year"):
            with self.subTest(question=question):
                self.assertTrue(
                    nlg.classify_followup(
                        question,
                        "How much is first-year BSIT tuition?",
                        "Please check with the Business Office.",
                    )
                )

    def test_context_does_not_treat_short_complete_questions_as_followups(self):
        context = ConversationContext("standalone-question-test")
        context.conversation_history.append({
            "user": "Is there school on Saturday?",
            "bot": "Saturday is a closed day.",
        })

        self.assertFalse(context.is_followup("Is PAC a Christian school?"))
        self.assertFalse(context.is_followup("BSN tuition"))
        self.assertTrue(context.is_followup("How about the second year?"))


class TuitionSpecificityTests(unittest.TestCase):
    def test_program_year_and_semester_must_exist_in_source(self):
        requirements = faq._tuition_requirements(
            "How much is tuition for BSBA first year second semester?"
        )

        self.assertTrue(requirements)
        self.assertFalse(
            faq._matches_tuition_requirements(
                "Tuition fees vary by program and year level.", requirements
            )
        )
        self.assertTrue(
            faq._matches_tuition_requirements(
                "BSBA Business Administration 1st Year 2nd Semester tuition: PHP 12,000",
                requirements,
            )
        )

    def test_course_only_question_still_requires_matching_program(self):
        requirements = faq._tuition_requirements("pila ang tuition fee sa nursing")

        self.assertFalse(
            faq._matches_tuition_requirements(
                "BSBA Business Administration tuition rates vary.", requirements
            )
        )
        self.assertTrue(
            faq._matches_tuition_requirements(
                "BSN Nursing tuition rates vary.", requirements
            )
        )


class SourceArbitrationTests(unittest.TestCase):
    def test_source_judge_receives_faq_handbook_and_date_evidence(self):
        handbook_chunks = [{
            "section_title": "Multifaith Services",
            "content": "PAC is a Seventh-day Adventist institution.",
        }]
        faq_candidates = [{
            "faq": (
                "PAC promotes Christian education.",
                "What is the vision of PAC?",
            ),
        }]
        date_answer = "On Saturday, all departments are closed."
        prompt_seen = []

        def choose_source(prompt, max_tokens):
            prompt_seen.append(prompt)
            return "DATES"

        result = faq._ai_pick_best_source(
            "Is there school on Saturday?",
            handbook_chunks,
            faq_candidates,
            choose_source,
            date_excerpt=date_answer,
        )

        self.assertEqual(result, "date")
        self.assertIn("--- HANDBOOK EXCERPTS ---", prompt_seen[0])
        self.assertIn("--- FAQ EXCERPTS ---", prompt_seen[0])
        self.assertIn("--- DYNAMIC DATE/SCHEDULE ANSWER ---", prompt_seen[0])

    def test_source_judge_abstains_on_unrecognized_verdict(self):
        self.assertEqual(
            faq._ai_pick_best_source(
                "Is there school on Saturday?",
                [{"section_title": "School schedule", "content": "School days"}],
                [],
                lambda prompt, max_tokens: "I am not sure",
                date_excerpt="Saturday is closed.",
            ),
            "neither",
        )

    def test_handbook_routes_religion_and_shorts_to_specific_evidence(self):
        with patch.object(handbook.ai_cache, "get", return_value=None), \
             patch.object(handbook, "llm_chat", return_value="YES"):
            self.assertTrue(handbook.is_handbook_question("What is the religion of PAC?"))
        self.assertEqual(
            handbook._forced_topic_for("What is the religion of PAC?")[0],
            ["multifaith services"],
        )
        short_topics = handbook._forced_topic_for(
            "Is it okay to wear short inside the campus?"
        )[0]
        self.assertIn("dress and appearance", short_topics)
        self.assertIn("first-degree offense", short_topics)
        self.assertFalse(
            handbook.is_handbook_question(
                "How much is the tuition fee for BSBA?"
            )
        )

    def test_handbook_classifier_uses_ai_and_caches_verdict_by_language(self):
        question = "Who founded the school?"
        with patch.object(handbook.ai_cache, "get", side_effect=[None, True]), \
             patch.object(handbook.ai_cache, "set") as cache_set, \
             patch.object(handbook, "llm_chat", return_value="YES.") as router:
            self.assertTrue(handbook.is_handbook_question(question, "english"))
            self.assertTrue(handbook.is_handbook_question(question, "english"))

        router.assert_called_once()
        cache_set.assert_called_once()


if __name__ == "__main__":
    unittest.main()
