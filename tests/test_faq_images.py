import unittest
from unittest.mock import patch

from routes.chat import _is_image_followup, _should_show_image
from services import faq
from services.conversation import ConversationContext, _apply_dict_to_ctx, _ctx_to_dict


class FaqImageAssociationTests(unittest.TestCase):
    def test_exact_faq_uses_its_own_image_not_another_faq_image(self):
        question = "Can I see the admissions application map?"
        answer = "The admissions office is in the main building."
        target_faq = (
            answer, question, 17, "Admissions", "application",
            None, None, 10, "uploads/admissions-map.png", None,
        )
        unrelated_faq = (
            "The library is on the second floor.",
            "Where is the library?", 18, "Library", "location",
            None, None, 10, "uploads/library-map.png", None,
        )
        ctx = ConversationContext("faq-image-test")
        ctx.last_detected_lang = "english"

        with patch.object(faq, "get_conversation_context", return_value=ctx), \
                patch.object(faq, "_get_all_faqs", return_value=[
                    target_faq, unrelated_faq,
                ]), \
                patch(
                    "services.corrections.get_exact_correction",
                    return_value=None,
                ), \
                patch.object(
                    faq.handbook_service,
                    "get_exact_handbook_answer",
                    return_value=None,
                ), \
                patch.object(
                    faq, "_validated_faq_source_answer", return_value=answer
                ), \
                patch.object(
                    faq.lang_service,
                    "classify_and_detect_language",
                ):
            result = faq.find_answer(question, return_topic=True)

        self.assertIn("[Image: uploads/admissions-map.png]", result[0])
        self.assertNotIn("library-map.png", result[0])

    def test_fuzzy_match_only_uses_image_when_faq_confidence_is_high(self):
        row = (
            "The admissions office is in the main building.",
            "Where is admissions?", 17, "Admissions", "location",
            None, None, 0, "uploads/admissions-map.png", None,
        )

        self.assertEqual(
            faq._image_for_faq_candidate({"combined": 0.9, "faq": row}),
            "uploads/admissions-map.png",
        )
        self.assertIsNone(
            faq._image_for_faq_candidate({"combined": 0.7, "faq": row}),
        )

    def test_close_faq_matches_with_different_images_suppress_image(self):
        first = {
            "combined": 0.91,
            "faq": ("Answer A", "Question A", 1, "Admissions", "", "", "", 0,
                    "uploads/admissions.png", None),
        }
        second = {
            "combined": 0.89,
            "faq": ("Answer B", "Question B", 2, "Library", "", "", "", 0,
                    "uploads/library.png", None),
        }

        self.assertIsNone(
            faq._image_for_faq_candidate(first, [first, second]),
        )

    def test_duplicate_exact_questions_with_different_images_suppress_image(self):
        question = "Where is the student services office?"
        rows = [
            ("Visit the student services office.", question, 1, "Services",
             "location", None, None, 10, "uploads/services-office.png", None),
            ("Visit the student services office.", question, 2, "Services",
             "location", None, None, 5, "uploads/old-office.png", None),
        ]

        with patch.object(faq, "_get_all_faqs", return_value=rows):
            match = faq._find_exact_faq(question)

        self.assertIsNotNone(match)
        self.assertIsNone(match[8])

    def test_chat_displays_an_associated_faq_image_without_an_extra_request(self):
        answer = "The admissions office is in the main building. " * 3

        self.assertFalse(_should_show_image(
            "Where is the admissions office?",
            answer,
        ))
        self.assertTrue(_should_show_image(
            "Can you show me the admissions office?",
            answer,
        ))
        self.assertTrue(_should_show_image(
            "Where is the admissions office?",
            f"{answer}\n\n[Image: uploads/admissions-map.png]",
        ))

    def test_chat_shows_contextual_faq_image_on_a_followup_request(self):
        answer = "[Image: https://res.cloudinary.com/demo/image/upload/president.jpg]"

        self.assertTrue(_is_image_followup("show me an image"))
        self.assertTrue(_is_image_followup("Can you show me a picture, please?"))
        self.assertTrue(_should_show_image("show me an image", answer, 3))
        self.assertFalse(_is_image_followup("who is the college president?"))
        self.assertFalse(_is_image_followup("show me an image of the campus"))

    def test_contextual_faq_image_survives_reload_but_not_another_answer(self):
        context = ConversationContext("faq-image-context")
        context.update(
            faq_answer="The president is Dr. Macias.",
            faq_question="Who is the college president?",
            faq_image="https://res.cloudinary.com/demo/image/upload/president.jpg",
        )
        restored = ConversationContext("faq-image-context")
        _apply_dict_to_ctx(restored, _ctx_to_dict(context))

        self.assertEqual(restored.last_faq_image_question, "Who is the college president?")
        self.assertEqual(
            restored.last_faq_image,
            "https://res.cloudinary.com/demo/image/upload/president.jpg",
        )

        restored.update(user_msg="Thanks", bot_response="You're welcome.")
        self.assertIsNone(restored.last_faq_image_question)
        self.assertIsNone(restored.last_faq_image)


if __name__ == "__main__":
    unittest.main()
