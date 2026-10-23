import unittest
from unittest.mock import patch

from flask import Flask

from routes.admin import bp


class AdminCorrectionRouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.secret_key = "test-secret"
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_invalid_faq_id_returns_bad_request_without_saving(self):
        with patch("routes.admin.admin_required", return_value=True), \
                patch("services.corrections.save_correction") as save_correction:
            response = self.client.post(
                "/api/admin/add_correction",
                json={
                    "question": "What is the dress code?",
                    "answer": "Students should dress modestly.",
                    "faq_id": "not-an-id",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "faq_id must be an integer"})
        save_correction.assert_not_called()

    def test_non_text_correction_fields_return_bad_request(self):
        with patch("routes.admin.admin_required", return_value=True), \
                patch("services.corrections.save_correction") as save_correction:
            response = self.client.post(
                "/api/admin/add_correction",
                json={"question": ["not text"], "answer": "A valid answer."},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {
            "error": "question and answer must be text",
        })
        save_correction.assert_not_called()

    def test_add_correction_button_request_saves_and_returns_success_shape(self):
        with patch("routes.admin.admin_required", return_value=True), \
                patch(
                    "services.corrections.save_correction",
                    return_value=True,
                ) as save_correction:
            response = self.client.post(
                "/api/admin/add_correction",
                json={
                    "question": " What is the dress code? ",
                    "answer": " Students should dress modestly. ",
                    "faq_id": "17",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"success": True})
        self.assertEqual(
            save_correction.call_args.args[1:3],
            ("What is the dress code?", "Students should dress modestly."),
        )
        self.assertEqual(save_correction.call_args.kwargs["faq_id"], 17)

    def test_single_delete_button_calls_delete_endpoint(self):
        with patch("routes.admin.admin_required", return_value=True), \
                patch(
                    "services.corrections.delete_correction",
                    return_value=True,
                ) as delete_correction:
            response = self.client.delete("/api/admin/corrections/24")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"success": True})
        self.assertEqual(delete_correction.call_args.args[1], 24)

    def test_bulk_delete_button_returns_deleted_ids(self):
        with patch("routes.admin.admin_required", return_value=True), \
                patch(
                    "services.corrections.delete_correction",
                    side_effect=lambda get_db, correction_id: correction_id == 24,
                ) as delete_correction:
            response = self.client.post(
                "/api/admin/corrections/bulk_delete",
                json={"ids": [24, 25, "invalid"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "success": True,
            "deleted": [24],
            "count": 1,
        })
        self.assertEqual(delete_correction.call_count, 2)

    def test_delete_correction_endpoints_reject_non_admins(self):
        with patch("routes.admin.admin_required", return_value=False), \
                patch("services.corrections.delete_correction") as delete_correction:
            single = self.client.delete("/api/admin/corrections/24")
            bulk = self.client.post(
                "/api/admin/corrections/bulk_delete",
                json={"ids": [24]},
            )

        self.assertEqual(single.status_code, 403)
        self.assertEqual(bulk.status_code, 403)
        delete_correction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
