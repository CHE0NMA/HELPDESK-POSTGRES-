import unittest
from unittest.mock import ANY, patch

from flask import Flask

from routes.chat import _chat_source_ref, _feedback_source, _log_chat, bp
from services import answer_memory


class FakeCursor:
    def __init__(self, row=None):
        self.query = ""
        self.params = ()
        self.executions = []
        self.row = row
        self.closed = False

    def execute(self, query, params=None):
        self.query = query
        self.params = params or ()
        self.executions.append((query, self.params))

    def fetchone(self):
        return self.row

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, row=None):
        self.cur = FakeCursor(row=row)
        self.committed = False
        self.closed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


class ChatFeedbackTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.secret_key = "test-secret"
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_unverified_answer_cache_entries_are_never_returned(self):
        connection = FakeConnection()
        cached = answer_memory.get_cached_answer(
            lambda: connection,
            "Can I wear shorts?",
            detected_lang="bisaya",
        )

        self.assertIsNone(cached)
        self.assertIn("is_verified = TRUE", connection.cur.query)
        self.assertEqual(connection.cur.params[1], "bisaya")

    def test_cached_answers_default_to_unverified(self):
        connection = FakeConnection()
        saved = answer_memory.store_successful_answer(
            lambda: connection,
            "Can I wear shorts?",
            "Students should follow the dress code policy.",
            detected_lang="bisaya",
        )

        self.assertTrue(saved)
        self.assertIn("is_verified", connection.cur.query)
        self.assertFalse(connection.cur.params[5])

    def test_feedback_source_is_resolved_from_matching_chat_history(self):
        connection = FakeConnection(
            row=("handbook_query", None, "Dress and appearance; Jewelry")
        )
        with patch("routes.chat.get_db", return_value=connection):
            source_type, source_ref, faq_id = _feedback_source(
                "session-1",
                "unsa pwede mag earrings?",
                "Dili pwede.",
            )

        self.assertEqual((source_type, source_ref, faq_id), (
            "handbook",
            "Dress and appearance; Jewelry",
            None,
        ))
        self.assertIn("source_ref", connection.cur.query)
        self.assertIn("FROM chat_history", connection.cur.query)
        self.assertEqual(
            connection.cur.params,
            ("session-1", "unsa pwede mag earrings?", "Dili pwede."),
        )

    def test_chat_history_retains_faq_and_handbook_source_references(self):
        self.assertEqual(_chat_source_ref("s1", "faq_query", faq_id=7), "FAQ #7")
        with patch(
            "routes.chat.get_conversation_context",
            return_value=type(
                "Context",
                (),
                {"handbook_sections": ["Dress and appearance", "Jewelry"]},
            )(),
        ):
            source_ref = _chat_source_ref("s1", "handbook_query")
        self.assertEqual(source_ref, "Dress and appearance; Jewelry")

        connection = FakeConnection()
        with self.client.application.test_request_context("/"), \
             patch("routes.chat.get_db", return_value=connection):
            _log_chat(
                "session-1",
                "Question?",
                "Answer.",
                "handbook_query",
                "bisaya",
                source_ref=source_ref,
            )
        self.assertIn("source_ref", connection.cur.query)
        self.assertEqual(connection.cur.params[7], source_ref)

    def test_recorded_feedback_keeps_language_and_deletes_only_matching_cache(self):
        connection = FakeConnection()
        saved = answer_memory.record_feedback(
            lambda: connection,
            "pila ang tuition?",
            "thumbs_down",
            source_type="faq",
            faq_id=8,
            session_id="session-1",
            detected_lang="bisaya",
        )

        self.assertTrue(saved)
        insert_query, insert_params = connection.cur.executions[0]
        delete_query, delete_params = connection.cur.executions[1]
        self.assertIn("detected_language", insert_query)
        self.assertEqual(insert_params[6:9], (8, "bisaya", "session-1"))
        self.assertIn("detected_lang = %s", delete_query)
        self.assertEqual(delete_params, ("pila ang tuition", "bisaya"))

    def test_students_cannot_submit_answer_corrections(self):
        with patch("routes.chat.get_db") as get_db:
            response = self.client.post(
                "/api/satisfaction",
                json={
                    "rating": "thumbs_down",
                    "question": "Can I wear earrings?",
                    "answer": "Yes.",
                    "correction": "Students may not wear earrings.",
                },
            )

        self.assertEqual(response.status_code, 403)
        get_db.assert_not_called()

    def test_thumbs_down_without_correction_is_still_recorded(self):
        connection = FakeConnection()
        with patch("routes.chat.get_db", return_value=connection), \
             patch("routes.chat.lang_service.detect_language", return_value="english"), \
             patch("routes.chat.lang_service.classify_topic", return_value="dress_code"), \
             patch("services.answer_memory.record_feedback", return_value=True) as record_feedback, \
             patch("routes.chat.log_unanswered_standalone"):
            response = self.client.post(
                "/api/satisfaction",
                json={
                    "rating": "thumbs_down",
                    "question": "Can I wear earrings?",
                    "answer": "Yes.",
                    "correction": "",
                    "session_id": "test-session",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "success")
        self.assertTrue(connection.committed)
        self.assertTrue(connection.closed)
        self.assertIn("detected_language", connection.cur.query)
        self.assertIn("detected_topic", connection.cur.query)
        self.assertIn("source_type", connection.cur.query)
        self.assertEqual(connection.cur.params[4:6], ("english", "dress_code"))
        self.assertEqual(record_feedback.call_args.kwargs["detected_lang"], "english")
        self.assertEqual(record_feedback.call_args.kwargs["source_type"], "unknown")
        self.assertNotIn("correction", record_feedback.call_args.kwargs)

    def test_thumbs_up_records_detected_language_and_persisted_faq_source(self):
        connection = FakeConnection()
        with self.client.session_transaction() as browser_session:
            browser_session["persistent_session_id"] = "server-session"
        with patch("routes.chat.get_db", return_value=connection), \
             patch("routes.chat.lang_service.detect_language", return_value="tagalog"), \
             patch("routes.chat.lang_service.classify_topic", return_value="dress_code"), \
             patch(
                 "routes.chat._feedback_source",
                 return_value=("faq", "FAQ #7", 7),
             ) as source_lookup, patch(
                 "services.answer_memory.record_feedback",
                 return_value=True,
             ) as record_feedback:
            response = self.client.post(
                "/api/satisfaction",
                json={
                    "rating": "thumbs_up",
                    "question": "pwede ba mag hikaw?",
                    "answer": "Dili pwede maghikaw.",
                    "session_id": "client-session",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(connection.cur.params[0], "server-session")
        self.assertEqual(source_lookup.call_args.args[0], "server-session")
        self.assertEqual(connection.cur.params[6:9], ("faq", "FAQ #7", 7))
        self.assertEqual(record_feedback.call_args.args[2], "thumbs_up")
        self.assertEqual(record_feedback.call_args.kwargs["detected_lang"], "tagalog")
        self.assertEqual(record_feedback.call_args.kwargs["faq_id"], 7)

    def test_thumbs_down_suggestion_enters_staff_review_queue(self):
        connection = FakeConnection()
        suggestion = "Students are not allowed to wear earrings at school."
        with patch("routes.chat.get_db", return_value=connection), \
             patch("routes.chat.lang_service.detect_language", return_value="english"), \
             patch("routes.chat.lang_service.classify_topic", return_value="dress_code"), \
             patch("routes.chat._feedback_source", return_value=("handbook", "dress policy", None)), \
             patch("services.answer_memory.record_feedback", return_value=True), \
             patch(
                 "services.self_heal.save_feedback_draft",
                 return_value=True,
             ) as save_draft, patch("routes.chat.log_unanswered_standalone"):
            response = self.client.post(
                "/api/satisfaction",
                json={
                    "rating": "thumbs_down",
                    "question": "Can I wear earrings?",
                    "answer": "Yes.",
                    "suggestion": suggestion,
                    "session_id": "test-session",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["correction_queued"])
        save_draft.assert_called_once_with(
            ANY,
            "Can I wear earrings?",
            suggestion,
            source_ref="dress policy",
            detected_language="english",
        )

    def test_report_feedback_records_language_and_topic(self):
        connection = FakeConnection()
        with patch("routes.chat.get_db", return_value=connection), \
             patch("routes.chat.lang_service.detect_language", return_value="bisaya"), \
             patch("routes.chat.lang_service.classify_topic", return_value="tuition"):
            response = self.client.post(
                "/feedback",
                json={
                    "question": "pila bayran sa nursing",
                    "answer": "I'm not sure.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("detected_language", connection.cur.query)
        self.assertEqual(connection.cur.params[-2:], ("bisaya", "tuition"))
        self.assertTrue(connection.committed)


if __name__ == "__main__":
    unittest.main()
