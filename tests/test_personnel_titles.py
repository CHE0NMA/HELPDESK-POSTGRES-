import unittest
from unittest.mock import patch

from services import faq
from services.answer_engine import get_top_faq_candidates
from services.conversation import ConversationContext
from services.language import _obviously_not_gibberish_and_school_related
from services.personnel_titles import personnel_titles


class PersonnelTitleTests(unittest.TestCase):
    def test_title_variants_resolve_to_the_same_personnel_role(self):
        examples = (
            ("Who is the College President?", "college_president"),
            ("Kinsa ang presidente sa eskwelahan?", "college_president"),
            ("Kinsa ang presidente sa kolehiyo?", "college_president"),
            ("Sino ang presidente ng kolehiyo?", "college_president"),
            ("Who is the VP - Finance?", "vp_finance"),
            ("Who is the Vice President for Finance?", "vp_finance"),
            ("Kinsa ang bise presidente sa pinansya?", "vp_finance"),
            ("Sino ang bise presidente ng pananalapi?", "vp_finance"),
            ("VP – ACADEMIC AFFAIRS", "vp_academic_affairs"),
            (
                "Kinsa ang bise presidente sa akademikong kalihokan?",
                "vp_academic_affairs",
            ),
            (
                "Sino ang bise presidente ng akademikong gawain?",
                "vp_academic_affairs",
            ),
            (
                "VP – STUDENT AFFAIRS & SERVICES/HRMO",
                "vp_student_affairs_services_hrmo",
            ),
            (
                "Kinsa ang bise presidente sa serbisyo sa estudyante?",
                "vp_student_affairs_services_hrmo",
            ),
            (
                "Sino ang bise presidente ng ugnayang pang estudyante?",
                "vp_student_affairs_services_hrmo",
            ),
            ("VP - DEVELOPMENT", "vp_development"),
            ("Kinsa ang bise presidente sa kalamboan?", "vp_development"),
            ("Sino ang bise presidente ng pagpapaunlad?", "vp_development"),
            ("Treasurer", "treasurer"),
            ("Kinsa ang tig tipig sa kwarta sa eskwelahan?", "treasurer"),
            ("Sino ang ingat-yaman ng paaralan?", "treasurer"),
            ("Accountant", "accountant"),
            ("Kinsa ang tig kwenta sa eskwelahan?", "accountant"),
            ("Sino ang accountant ng paaralan?", "accountant"),
            ("Director of Student Finance", "director_student_finance"),
            (
                "Kinsa ang direktor sa panalapi sa estudyante?",
                "director_student_finance",
            ),
            (
                "Sino ang direktor ng pananalapi ng estudyante?",
                "director_student_finance",
            ),
            ("Dean, Computer of Studies", "dean_computer_studies"),
            ("Kinsa ang dean sa computer studies?", "dean_computer_studies"),
            ("Kinsa gani ang dean sa computer studies?", "dean_computer_studies"),
            ("Sino ang dean ng computer studies?", "dean_computer_studies"),
            ("BSCS Coordinator", "bscs_coordinator"),
            ("Kinsa ang coordinator sa BSCS?", "bscs_coordinator"),
            ("Sino ang coordinator ng BSCS?", "bscs_coordinator"),
            ("BSIT Coordinator", "bsit_coordinator"),
            ("Kinsa ang coordinator sa BSIT?", "bsit_coordinator"),
            ("Sino ang coordinator ng BSIT?", "bsit_coordinator"),
        )

        for text, expected_role in examples:
            with self.subTest(text=text):
                self.assertIn(expected_role, personnel_titles(text))
                self.assertTrue(
                    _obviously_not_gibberish_and_school_related(text)
                )

    def test_title_query_prefers_faq_for_the_same_role(self):
        finance_faq = (
            "The VP-Finance is Alex.", "Who is the VP - Finance?",
            1, "Personnel", "VP-Finance", "finance", None, 0, None, None,
        )
        academic_faq = (
            "The VP-Academic Affairs is Jordan.",
            "Who is the VP - Academic Affairs?",
            2, "Personnel", "VP-Academic Affairs", "academic affairs",
            None, 0, None, None,
        )

        results = get_top_faq_candidates(
            "Who is the Vice President for Finance?",
            "Who is the Vice President for Finance?",
            [academic_faq, finance_faq],
        )

        self.assertEqual(results[0]["faq"][2], 1)
        self.assertTrue(results[0]["personnel_title_match"])

    def test_bisaya_president_question_prefers_college_president_faq(self):
        president_faq = (
            "The College President is Dr. Mariano Joaquin C. Macias Jr.",
            "Who is the College President?",
            1, "Administration", "college_president", None, None, 0, None, None,
        )
        history_faq = (
            "The college was founded in 1975.",
            "What is the history of the college?",
            2, "History", "college_history", None, None, 100, None, None,
        )

        questions = (
            (
                "kinsa ang college president",
                "kinsa ang college president who person in charge college president",
                "bisaya",
            ),
            (
                "kinsa ang college president sa school",
                "kinsa ang college president sa school who person in charge contact",
                "bisaya",
            ),
            (
                "who is the college president",
                "who is the college president",
                "english",
            ),
        )
        for question, expanded, language in questions:
            with self.subTest(question=question):
                results = get_top_faq_candidates(
                    question,
                    expanded,
                    [history_faq, president_faq],
                    detected_lang=language,
                )

                self.assertEqual(results[0]["faq"][2], 1)
                self.assertTrue(results[0]["personnel_title_match"])

    def test_bisaya_dean_query_uses_matching_personnel_faq_not_handbook(self):
        answer = (
            "The Dean of the College of Computer Studies is "
            "GLADYS MAE C. PIELAGO."
        )
        dean_faq = (
            answer,
            "Who is the Dean of Computer Studies?",
            7, "Personnel", "dean_computer_studies", None, None, 0, None, None,
        )
        candidate = {
            "faq": dean_faq,
            "combined": 0.8,
            "personnel_title_match": True,
        }
        context = ConversationContext("personnel-faq-test")

        with patch.object(faq, "_get_all_faqs", return_value=[dean_faq]), \
             patch.object(faq, "_get_faq_candidates", return_value=[candidate]), \
             patch.object(faq.circuit_breaker, "is_open", return_value=True):
            result = faq._answer_personnel_faq(
                "kinsa gani ang dean sa computer studies?",
                "bisaya",
                context,
            )

        self.assertEqual(
            result[0],
            faq.lang_service.get_translation_unavailable_response("bisaya"),
        )
        self.assertEqual(result[2], 7)
        self.assertIsNone(context.last_faq_question)


if __name__ == "__main__":
    unittest.main()
