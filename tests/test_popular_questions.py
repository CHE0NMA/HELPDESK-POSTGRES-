import unittest
from unittest.mock import patch

from flask import Flask
from routes.api import bp as api_bp
from routes.chat import bp as chat_bp
from services.popular_questions import get_popular_knowledge_questions


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = ()
        self.closed = False

    def execute(self, query, params=()):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, rows):
        self.cur = FakeCursor(rows)
        self.closed = False

    def cursor(self):
        return self.cur

    def close(self):
        self.closed = True


class PopularQuestionTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.secret_key = "test-secret"
        app.register_blueprint(chat_bp)
        app.register_blueprint(api_bp)
        self.client = app.test_client()

    def test_most_asked_endpoints_use_filtered_knowledge_questions(self):
        questions = [{
            "question": "How much is the tuition fee for BSCS?",
            "user_message": "How much is the tuition fee for BSCS?",
            "frequency": 12,
            "count": 12,
            "category": "Fees",
        }]
        with patch(
            "services.popular_questions.get_popular_knowledge_questions",
            return_value=questions,
        ) as get_popular:
            chat_response = self.client.get("/suggested_questions")
            featured_response = self.client.get("/api/featured")

        self.assertEqual(chat_response.get_json(), questions)
        self.assertEqual(featured_response.get_json(), questions)
        self.assertEqual(get_popular.call_count, 2)
        self.assertEqual(
            get_popular.call_args_list[0].kwargs,
            {"limit": 10, "days": 30},
        )
        self.assertEqual(
            get_popular.call_args_list[1].kwargs,
            {"limit": 8, "days": 60},
        )

    def test_popular_questions_only_return_canonical_faq_prompts(self):
        connection = FakeConnection([
            ("How much is the tuition fee for BSCS?", 12, "Fees"),
            ("Thanks", 8, "FAQ"),
            ("What is the tuition fee?", 7, "Fees"),
        ])

        items = get_popular_knowledge_questions(
            lambda: connection,
            limit=10,
            days=30,
        )

        self.assertEqual(
            [item["question"] for item in items],
            [
                "How much is the tuition fee for BSCS?",
                "What is the tuition fee?",
            ],
        )
        self.assertTrue(all(item["category"] for item in items))
        self.assertEqual(connection.cur.params, (30, 50))
        self.assertIn("LEFT JOIN faq f ON f.id = ch.faq_id", connection.cur.query)
        self.assertIn("WHERE f.id IS NOT NULL", connection.cur.query)
        self.assertNotIn("handbook_query", connection.cur.query)
        self.assertTrue(connection.cur.closed)
        self.assertTrue(connection.closed)

    def test_related_suggestions_are_always_canonical_faq_questions(self):
        class SuggestionCursor(FakeCursor):
            def __init__(self):
                super().__init__([("What are the admission requirements?",)])
                self.queries = []

            def execute(self, query, params=()):
                super().execute(query, params)
                self.queries.append(query)

        class SuggestionConnection(FakeConnection):
            def __init__(self):
                self.cur = SuggestionCursor()
                self.closed = False

        connection = SuggestionConnection()
        with patch("routes.chat.get_db", return_value=connection):
            response = self.client.post(
                "/related_suggestions",
                json={
                    "user_message": "offensive text from a chat turn",
                    "category": "Admissions",
                },
            )

        self.assertEqual(response.get_json(), [
            {"user_message": "What are the admission requirements?"},
        ])
        self.assertTrue(connection.cur.queries)
        self.assertTrue(all("JOIN faq f ON f.id = ch.faq_id" in q for q in connection.cur.queries))
        self.assertTrue(all("f.question" in q for q in connection.cur.queries))

    def test_reuses_caller_connection_for_dashboard_analytics(self):
        connection = FakeConnection([
            ("What are the admission requirements for BSIT?", 4, "Admissions"),
        ])

        items = get_popular_knowledge_questions(
            lambda: self.fail("Should reuse the open connection"),
            limit=20,
            days=None,
            connection=connection,
        )

        self.assertEqual(items[0]["count"], 4)
        self.assertEqual(connection.cur.params, (100,))
        self.assertFalse(connection.closed)
        self.assertTrue(connection.cur.closed)

    def test_limits_are_bounded_and_contextless_phrases_are_excluded(self):
        connection = FakeConnection([
            ("Then?", 100, "Handbook"),
            ("What is the campus location?", 2, "FAQ"),
        ])

        items = get_popular_knowledge_questions(
            lambda: connection,
            limit=1000,
            days=10000,
        )

        self.assertEqual([item["question"] for item in items], [
            "What is the campus location?",
        ])
        self.assertEqual(connection.cur.params, (3650, 500))


if __name__ == "__main__":
    unittest.main()
