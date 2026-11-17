import unittest
from unittest.mock import Mock, patch

from flask import Flask

from routes.admin import bp


class FaqAddRouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.secret_key = "test-secret"
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_add_faq_saves_and_returns_success(self):
        conn = Mock()
        cur = conn.cursor.return_value
        with patch("routes.admin.admin_required", return_value=True), \
                patch("routes.admin.check_duplicate_faq", return_value=[]), \
                patch("routes.admin.get_db", return_value=conn), \
                patch("routes.admin.faq_cache.invalidate"), \
                patch("routes.admin.invalidate_response_tables"), \
                patch("routes.admin.invalidate_dashboard_cache"), \
                patch("services.embeddings.store_embedding_for_faq"):
            response = self.client.post(
                "/add_faq",
                data={
                    "topic": "General",
                    "question": "How do I contact support?",
                    "answer": "Contact the help desk.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        self.assertEqual(response.get_json()["message"], "FAQ added successfully")
        self.assertIn("INSERT INTO faq", cur.execute.call_args.args[0])
        conn.commit.assert_called_once()

    def test_database_error_during_duplicate_check_returns_json_error(self):
        with patch("routes.admin.admin_required", return_value=True), \
                patch(
                    "routes.admin.check_duplicate_faq",
                    side_effect=RuntimeError("database unavailable"),
                ):
            response = self.client.post(
                "/add_faq",
                data={"question": "How do I contact support?"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "database unavailable")


if __name__ == "__main__":
    unittest.main()
