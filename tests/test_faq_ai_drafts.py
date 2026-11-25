import json
import unittest
from unittest.mock import Mock, patch

from flask import Flask

from routes.admin import bp
from services import handbook_ingest


class GenerateFaqMetadataTests(unittest.TestCase):
    def test_metadata_generation_normalizes_alternate_questions(self):
        response = {
            "question": "How do I request a transcript?",
            "related_questions": ["Where can I get my TOR?", "Transcript request"],
            "topic": "Registrar",
            "intent": "transcript_request",
            "keywords": "transcript, TOR, registrar",
            "language": "en",
            "category": "Records",
        }
        with patch.object(handbook_ingest, "HAS_AI", True), patch.object(
            handbook_ingest, "call_for_ingest", return_value=json.dumps(response)
        ) as call:
            draft = handbook_ingest.generate_faq_metadata(
                "Submit a request to the Registrar to receive your transcript."
            )

        self.assertEqual(draft["question"], response["question"])
        self.assertEqual(
            draft["related_questions"],
            "Where can I get my TOR?|Transcript request",
        )
        self.assertEqual(draft["topic"], "Registrar")
        self.assertIn("using only the answer below", call.call_args.args[0])

    def test_metadata_generation_reports_unconfigured_ai(self):
        with patch.object(handbook_ingest, "HAS_AI", False):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                handbook_ingest.generate_faq_metadata("An answer.")

    def test_section_generation_includes_metadata_fields(self):
        generated = {
            "faqs": [{
                "question": "Can students wear shorts?",
                "related_questions": ["Are shorts allowed on campus?"],
                "answer": "Short pants are not allowed on campus.",
                "topic": "Dress and Appearance",
                "intent": "dress_code",
                "keywords": "short pants, dress code",
                "language": "en",
                "priority": 8,
            }]
        }
        with patch.object(handbook_ingest, "HAS_AI", True), patch.object(
            handbook_ingest, "call_for_ingest", return_value=json.dumps(generated)
        ):
            faqs = handbook_ingest.section_to_faqs({
                "section_title": "Dress and Appearance",
                "content": "Short pants are not allowed on campus. " * 4,
            })

        self.assertEqual(len(faqs), 1)
        self.assertEqual(faqs[0]["related_questions"], "Are shorts allowed on campus?")
        self.assertEqual(faqs[0]["intent"], "dress_code")
        self.assertEqual(faqs[0]["topic"], "Dress and Appearance")

    def test_section_generation_splits_pipe_separated_alternates_only(self):
        generated = {
            "faqs": [{
                "question": "Can students wear shorts?",
                "related_questions": "Are shorts allowed?|What is the rule?",
                "answer": "Short pants are not allowed on campus.",
                "keywords": "short pants, dress code",
            }]
        }
        with patch.object(handbook_ingest, "HAS_AI", True), patch.object(
            handbook_ingest, "call_for_ingest", return_value=json.dumps(generated)
        ):
            faqs = handbook_ingest.section_to_faqs({
                "section_title": "Dress and Appearance",
                "content": "Short pants are not allowed on campus. " * 4,
            })

        self.assertEqual(
            faqs[0]["related_questions"],
            "Are shorts allowed?|What is the rule?",
        )

    def test_section_generation_retries_invalid_json_and_preserves_bisaya_aliases(self):
        generated = {
            "faqs": [{
                "question": "How do I request a transcript?",
                "related_questions": ["Unsaon pagkuha sa transcript?"],
                "answer": "Submit a request to the Registrar for a transcript.",
                "keywords": ["transcript", "registrar"],
            }]
        }
        with patch.object(handbook_ingest, "HAS_AI", True), patch.object(
            handbook_ingest,
            "call_for_ingest",
            side_effect=["not json", json.dumps(generated)],
        ) as call, patch.object(handbook_ingest.time, "sleep"):
            faqs = handbook_ingest.section_to_faqs({
                "section_title": "Transcript requests",
                "content": "Submit a request to the Registrar for a transcript. " * 3,
            })

        self.assertEqual(call.call_count, 2)
        self.assertIn("Bisaya/Cebuano", call.call_args.args[0])
        self.assertEqual(faqs[0]["related_questions"], "Unsaon pagkuha sa transcript?")
        self.assertEqual(faqs[0]["keywords"], "transcript, registrar")

    def test_section_generation_reports_missing_ai_configuration(self):
        with patch.object(handbook_ingest, "HAS_AI", False):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                handbook_ingest.section_to_faqs({
                    "section_title": "Transcript requests",
                    "content": "Useful handbook content. " * 5,
                })

    def test_pasted_json_preserves_metadata_and_related_question_lists(self):
        faqs = handbook_ingest.parse_pasted_faqs(json.dumps({
            "faqs": [{
                "question": "How do I apply?",
                "answer": "Submit an application.",
                "topic": "Admissions",
                "intent": "apply",
                "related_questions": ["How can I enroll?", "Application process?"],
                "language": "en",
                "category": "Enrollment",
            }]
        }))

        self.assertEqual(faqs[0]["topic"], "Admissions")
        self.assertEqual(faqs[0]["intent"], "apply")
        self.assertEqual(
            faqs[0]["related_questions"],
            "How can I enroll?|Application process?",
        )
        self.assertEqual(faqs[0]["category"], "Enrollment")


class FaqDraftRoutesTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.secret_key = "test-secret"
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_add_faq_ai_endpoint_returns_reviewable_draft(self):
        draft = {
            "question": "How do I contact support?",
            "related_questions": "Where can I get help?",
            "topic": "Help Desk",
            "intent": "contact_support",
            "keywords": "support, contact",
            "language": "en",
            "category": "General",
        }
        with patch("routes.admin.admin_required", return_value=True), patch(
            "services.handbook_ingest.generate_faq_metadata", return_value=draft
        ):
            response = self.client.post(
                "/api/admin/generate_faq_draft",
                json={"answer": "Contact the help desk for support."},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["draft"], draft)

    def test_add_faq_ai_endpoint_requires_answer(self):
        with patch("routes.admin.admin_required", return_value=True):
            response = self.client.post("/api/admin/generate_faq_draft", json={})

        self.assertEqual(response.status_code, 400)

    def test_handbook_preview_returns_generated_faqs_without_inserting(self):
        conn = Mock()
        conn.cursor.return_value.fetchone.return_value = (
            "Student Conduct", "Dress and Appearance", 20, 21, "Handbook content"
        )
        drafts = [{
            "question": "What is the dress policy?",
            "answer": "Dress modestly.",
            "topic": "Dress",
            "intent": "dress_code",
            "keywords": "dress, attire",
            "related_questions": "What should I wear?",
            "language": "en",
            "priority": 5,
        }]
        with patch("routes.admin.admin_required", return_value=True), patch(
            "routes.admin.get_db", return_value=conn
        ), patch("services.handbook_ingest.section_to_faqs", return_value=drafts):
            response = self.client.post(
                "/api/admin/handbook_chunks/7/generate_faqs"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["faqs"], drafts)
        conn.commit.assert_not_called()

    def test_handbook_preview_reports_unavailable_ai_as_retryable(self):
        conn = Mock()
        conn.cursor.return_value.fetchone.return_value = (
            "Student Conduct", "Dress and Appearance", 20, 21, "Handbook content"
        )
        with patch("routes.admin.admin_required", return_value=True), patch(
            "routes.admin.get_db", return_value=conn
        ), patch(
            "services.handbook_ingest.section_to_faqs",
            side_effect=RuntimeError("AI providers did not return a response."),
        ):
            response = self.client.post(
                "/api/admin/handbook_chunks/7/generate_faqs"
            )

        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.get_json()["retryable"])

    def test_handbook_preview_returns_empty_result_without_http_error(self):
        conn = Mock()
        conn.cursor.return_value.fetchone.return_value = (
            "Student Conduct", "Dress and Appearance", 20, 21, "Useful content " * 10
        )
        with patch("routes.admin.admin_required", return_value=True), patch(
            "routes.admin.get_db", return_value=conn
        ), patch(
            "services.handbook_ingest.section_to_faqs", return_value=[]
        ):
            response = self.client.post(
                "/api/admin/handbook_chunks/7/generate_faqs"
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        self.assertFalse(response.get_json()["faqs"])
        self.assertTrue(response.get_json()["retryable"])

    def test_reviewed_handbook_insert_does_not_retire_existing_faqs(self):
        reviewed = [{
            "question": "What is the dress policy?",
            "answer": "Dress modestly.",
            "topic": "Dress",
            "intent": "dress_code",
            "keywords": "dress, attire",
            "related_questions": "What should I wear?",
            "language": "en",
            "priority": 7,
        }]
        with patch("routes.admin.admin_required", return_value=True), patch(
            "routes.admin.mass_insert_faqs", return_value=1
        ) as insert, patch("routes.admin.faq_cache.invalidate"), patch(
            "routes.admin.invalidate_response_tables"
        ), patch("routes.admin.invalidate_dashboard_cache"):
            response = self.client.post(
                "/api/admin/handbook_faqs", json={"faqs": reviewed}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["inserted"], 1)
        self.assertFalse(insert.call_args.kwargs["retire_previous"])


if __name__ == "__main__":
    unittest.main()
