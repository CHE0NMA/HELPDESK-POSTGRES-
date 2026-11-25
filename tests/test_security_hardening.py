import unittest
from unittest.mock import patch

from flask import Flask

from extensions import csrf
from routes.admin import bp as admin_bp
from routes import cron
from utils.security import client_ip


class SecurityHardeningTests(unittest.TestCase):
    def test_client_ip_ignores_untrusted_forwarded_header(self):
        app = Flask(__name__)
        with app.test_request_context(
            "/",
            headers={"X-Forwarded-For": "203.0.113.99"},
            environ_base={"REMOTE_ADDR": "192.0.2.10"},
        ):
            self.assertEqual(client_ip(), "192.0.2.10")

    def test_cron_secret_is_only_accepted_as_bearer_header(self):
        app = Flask(__name__)
        with patch.object(cron, "CRON_SECRET", "test-secret"):
            with app.test_request_context("/job?secret=test-secret"):
                self.assertFalse(cron._authorized())
            with app.test_request_context(
                "/job",
                headers={"Authorization": "Bearer test-secret"},
            ):
                self.assertTrue(cron._authorized())

    def test_cron_is_disabled_without_a_secret_in_vercel_environment(self):
        app = Flask(__name__)
        with patch.object(cron, "CRON_SECRET", ""), \
             patch.dict("os.environ", {"VERCEL_ENV": "production"}, clear=False):
            with app.test_request_context("/job"):
                self.assertFalse(cron._authorized())

    def test_password_change_requires_a_csrf_token(self):
        app = Flask(__name__)
        app.config.update(SECRET_KEY="test-secret", WTF_CSRF_ENABLED=True)
        csrf.init_app(app)
        app.register_blueprint(admin_bp)

        response = app.test_client().post("/api/admin/profile/password")

        self.assertEqual(response.status_code, 400)
        self.assertIn("CSRF", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
