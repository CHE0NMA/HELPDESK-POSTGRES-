import unittest

from services.formatting import format_bot_message


class BotMessageFormattingTests(unittest.TestCase):
    def test_formats_paragraphs_headings_and_lists(self):
        answer = (
            "## Next steps\n"
            "Submit your form.\n\n"
            "1. Check your account.\n"
            "2. Contact the office.\n\n"
            "- Bring your ID.\n"
            "- Bring your receipt."
        )

        formatted = format_bot_message(answer)

        self.assertIn("<h3>Next steps</h3>", formatted)
        self.assertIn("<p>Submit your form.</p>", formatted)
        self.assertIn("<ol><li>Check your account.</li><li>Contact the office.</li></ol>", formatted)
        self.assertIn("<ul><li>Bring your ID.</li><li>Bring your receipt.</li></ul>", formatted)

    def test_escapes_html_while_preserving_bold_and_italic(self):
        formatted = format_bot_message(
            "Use **bold** and *italic*. <script>alert('x')</script>"
        )

        self.assertIn("<strong>bold</strong>", formatted)
        self.assertIn("<em>italic</em>", formatted)
        self.assertIn("&lt;script&gt;", formatted)
        self.assertNotIn("<script>", formatted)

    def test_hides_images_when_requested(self):
        formatted = format_bot_message(
            "Details:\n[Image: uploads/sample.png]", show_images=False
        )

        self.assertIn("<p>Details:</p>", formatted)
        self.assertNotIn("<img", formatted)

    def test_chat_image_has_loading_placeholder_and_safe_source(self):
        formatted = format_bot_message("[Image: uploads/school-map.png]")

        self.assertIn("Loading attached image", formatted)
        self.assertIn('src="/uploads/school-map.png"', formatted)
        self.assertIn('alt="FAQ image"', formatted)
        self.assertIn('class="bot-image-frame"', formatted)


if __name__ == "__main__":
    unittest.main()
