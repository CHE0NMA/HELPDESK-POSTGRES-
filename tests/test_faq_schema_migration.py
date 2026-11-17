import unittest
from unittest.mock import Mock

from database.schema import _ensure_faq_editor_columns


class FaqEditorColumnMigrationTests(unittest.TestCase):
    def test_adds_columns_used_by_faq_editor_for_legacy_tables(self):
        cur = Mock()
        _ensure_faq_editor_columns(cur)

        statements = [call.args[0] for call in cur.execute.call_args_list]
        for column in (
            "intent VARCHAR(100) DEFAULT ''",
            "image_path VARCHAR(255)",
            "keywords TEXT",
            "related_questions TEXT",
            "language VARCHAR(20) DEFAULT 'en'",
            "category VARCHAR(100)",
            "priority_score INTEGER DEFAULT 0",
            "attachment_url VARCHAR(255)",
        ):
            self.assertIn(
                f"ALTER TABLE faq ADD COLUMN IF NOT EXISTS {column}",
                statements,
            )


if __name__ == "__main__":
    unittest.main()
