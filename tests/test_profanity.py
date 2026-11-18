import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from routes.chat import _flag_profanity
from services.profanity import find_flagged_terms


class ProfanityDetectionTests(unittest.TestCase):
    def test_detects_supplied_variants_case_insensitively_and_in_phrases(self):
        detected = find_flagged_terms(
            "PUTANG INA, g@go! They said pisteng-yawa and mukhang pera."
        )

        self.assertEqual(
            set(detected),
            {"putang ina", "g@go", "pisteng yawa", "yawa", "mukhang pera"},
        )

    def test_every_supplied_term_is_detectable(self):
        from services.profanity import PROFANITY_TERMS

        for term in PROFANITY_TERMS:
            with self.subTest(term=term):
                self.assertIn(term, find_flagged_terms(f"Please note: {term}."))

    def test_word_boundaries_do_not_flag_partial_matches(self):
        self.assertEqual(
            find_flagged_terms("The potato is in the animalistic illustration."),
            [],
        )

    def test_chat_still_answers_flagged_questions_with_a_warning(self):
        from routes.chat import bp
        from flask import Flask

        app = Flask(__name__)
        app.secret_key = "test-secret"
        app.register_blueprint(bp)
        client = app.test_client()
        with client.session_transaction() as browser_session:
            browser_session["persistent_session_id"] = "test-session"
            browser_session["last_activity"] = "2026-11-18T04:00:00"

        context = SimpleNamespace(
            last_detected_lang=None,
            last_faq_image=None,
            last_faq_image_question=None,
        )
        with patch("routes.chat._session_expired", return_value=False), \
                patch("routes.chat.check_session_rate_limit", return_value=(True, 1)), \
                patch("routes.chat.get_conversation_context", return_value=context), \
                patch("routes.chat.lang_service.detect_language", return_value="english"), \
                patch("routes.chat.find_answer", return_value=(
                    "Tuition payments are handled by Finance.",
                    "Tuition", 17, "tuition", "english",
                )), \
                patch("routes.chat.find_flagged_terms", return_value=["gago"]), \
                patch(
                    "routes.chat._flag_profanity",
                    return_value="Please keep your message respectful.",
                ) as flag_message, \
                patch("routes.chat._log_chat"), \
                patch("routes.chat.model_metrics.record_answer_timing"):
            response = client.post(
                "/chat",
                json={"message": "gago, how do I pay tuition?"},
            )

        self.assertEqual(response.status_code, 200)
        reply = response.get_json()["reply"]
        self.assertIn("Please keep your message respectful.", reply)
        self.assertIn("Tuition payments are handled by Finance.", reply)
        flag_message.assert_called_once()

    def test_flag_is_saved_to_existing_admin_reports_source(self):
        conn = Mock()
        cur = conn.cursor.return_value
        with patch("routes.chat.get_db", return_value=conn):
            warning = _flag_profanity(
                "gago, how do I pay tuition?",
                "english",
                ["gago"],
            )

        self.assertIn("respectful", warning)
        self.assertIn("INSERT INTO bad_feedback", cur.execute.call_args.args[0])
        self.assertEqual(cur.execute.call_args.args[1][2:], ("english", "profanity"))
        conn.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
