import unittest
from unittest.mock import patch

from flask import Flask

from routes.chat import bp


class FakeCursor:
    def __init__(self):
        self.query = ""
        self.params = ()
        self.closed = False

    def execute(self, query, params=None):
        self.query = query
        self.params = params or ()

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cur = FakeCursor()
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
        self.assertEqual(connection.cur.params[-2:], ("english", "handbook"))
        self.assertNotIn("correction", record_feedback.call_args.kwargs)

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
