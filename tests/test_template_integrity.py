import ast
import unittest
from pathlib import Path

from flask import Flask, render_template


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "templates"
ROUTE_FILES = (
    ROOT / "routes" / "pages.py",
    ROOT / "routes" / "admin.py",
    ROOT / "routes" / "schedules.py",
)


class TemplateIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = Flask(
            "template_integrity_tests",
            template_folder=str(TEMPLATE_ROOT),
        )
        cls.app.add_url_rule(
            "/", endpoint="pages.landing", view_func=lambda: "home"
        )
        cls.app.add_url_rule(
            "/new_chat", endpoint="pages.new_chat", view_func=lambda: "chat"
        )

    def test_all_html_templates_compile(self):
        templates = sorted(TEMPLATE_ROOT.rglob("*.html"))
        self.assertTrue(templates, "No HTML templates found")
        for path in templates:
            template_name = path.relative_to(TEMPLATE_ROOT).as_posix()
            with self.subTest(template=template_name):
                self.app.jinja_env.get_template(template_name)

    def test_route_template_references_resolve(self):
        references: list[tuple[str, str]] = []
        for route_file in ROUTE_FILES:
            tree = ast.parse(route_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "render_template"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    references.append((route_file.name, node.args[0].value))

        self.assertTrue(references, "No render_template references found")
        for source, template_name in references:
            with self.subTest(source=source, template=template_name):
                self.app.jinja_env.get_template(template_name)

    def test_templates_are_grouped(self):
        self.assertFalse(
            list(TEMPLATE_ROOT.glob("*.html")),
            "Move page templates into public/, admin/, or errors/",
        )
        self.assertTrue((TEMPLATE_ROOT / "errors" / "error.html").is_file())

    def test_error_page_renders_both_status_codes(self):
        with self.app.test_request_context("/"):
            for status_code, title in (
                (404, "This page is missing."),
                (500, "We hit a snag."),
            ):
                with self.subTest(status_code=status_code):
                    html = render_template(
                        "errors/error.html",
                        status_code=status_code,
                        title=title,
                        message="Please try again.",
                    )
                    self.assertIn(str(status_code), html)
                    self.assertIn(title, html)
                    self.assertIn('href="/"', html)
                    self.assertIn('href="/new_chat"', html)

    def test_admin_draft_page_displays_student_suggestions(self):
        draft: dict[str, object] = {
            "id": 7,
            "question": "Who founded the school?",
            "draft_answer": "Generated answer awaiting staff review.",
            "source_type": "student_feedback",
            "source_ref": "Verify before approving",
            "times_asked": 1,
            "student_suggestions": ["The student-proposed answer for staff review."],
        }
        with self.app.test_request_context("/"):
            html = render_template("admin/self_heal_review.html", drafts=[draft])

        self.assertIn("student-proposed answer", html)
        self.assertIn("draft-answer-7", html)
        self.assertIn("Use suggestion", html)


if __name__ == "__main__":
    unittest.main()
