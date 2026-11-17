import unittest

from services.answer_engine import get_top_faq_candidates
from services.language import _obviously_not_gibberish_and_school_related
from services.personnel_titles import personnel_titles


class PersonnelTitleTests(unittest.TestCase):
    def test_title_variants_resolve_to_the_same_personnel_role(self):
        examples = (
            ("Who is the VP - Finance?", "vp_finance"),
            ("Who is the Vice President for Finance?", "vp_finance"),
            ("VP – ACADEMIC AFFAIRS", "vp_academic_affairs"),
            (
                "VP – STUDENT AFFAIRS & SERVICES/HRMO",
                "vp_student_affairs_services_hrmo",
            ),
            ("VP - DEVELOPMENT", "vp_development"),
            ("Treasurer", "treasurer"),
            ("Accountant", "accountant"),
            ("Director of Student Finance", "director_student_finance"),
            ("Dean, Computer of Studies", "dean_computer_studies"),
            ("BSCS Coordinator", "bscs_coordinator"),
            ("BSIT Coordinator", "bsit_coordinator"),
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


if __name__ == "__main__":
    unittest.main()
