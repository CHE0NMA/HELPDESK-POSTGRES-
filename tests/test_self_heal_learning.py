import unittest
import sys
import types
from unittest.mock import Mock, patch

from services import self_heal


class FakeCursor:
    def __init__(self, row=None, rowcount=1):
        self.row = row
        self.rowcount = rowcount
        self.query: str = ""
        self.params: tuple = ()

    def execute(self, query, params=None):
        self.query = query
        self.params = params or ()

    def fetchone(self):
        return self.row

    def close(self):
        pass


class FakeConnection:
    def __init__(self, row=None, rowcount=1):
        self.cur = FakeCursor(row=row, rowcount=rowcount)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class FeedbackDraftTests(unittest.TestCase):
    def test_feedback_suggestion_is_queued_without_replacing_existing_draft(self):
        connection = FakeConnection()
        with patch.object(self_heal, "ensure_self_heal_schema", return_value=True), \
                patch.object(self_heal, "_normalize", return_value="school founder"):
            saved = self_heal.save_feedback_draft(
                lambda: connection,
                "Who founded the school?",
                "The Macias family founded the school in 1975.",
            )

        self.assertTrue(saved)
        self.assertTrue(connection.committed)
        self.assertIn("ON CONFLICT (question_norm) DO UPDATE", connection.cur.query)
        self.assertIn("student_suggestions", connection.cur.query)
        self.assertNotIn("draft_answer = EXCLUDED.draft_answer", connection.cur.query)
        self.assertEqual(connection.cur.params[2], "The Macias family founded the school in 1975.")
        self.assertTrue(connection.closed)

    def test_too_short_suggestion_is_not_queued(self):
        get_db = Mock(side_effect=AssertionError("DB should not be opened"))
        with patch.object(self_heal, "_normalize", return_value="question"):
            self.assertFalse(self_heal.save_feedback_draft(get_db, "Question?", "Too short"))
        get_db.assert_not_called()


class AdminApprovalTests(unittest.TestCase):
    def test_edited_admin_answer_is_the_answer_promoted(self):
        lookup = FakeConnection(row=("Who founded the school?", "Initial generated draft."))
        update = FakeConnection()
        get_db = Mock(side_effect=[lookup, update])
        reviewed_answer = "The school was founded by Mariano and Margie Macias in 1975."
        corrections = types.ModuleType("services.corrections")
        save_correction = Mock(return_value=True)
        setattr(corrections, "save_correction", save_correction)

        with patch.object(self_heal, "_normalize", return_value="who founded the school"), \
            patch.dict(sys.modules, {"services.corrections": corrections}):
            approved = self_heal.approve_draft(
                get_db,
                7,
                reviewed_by="staff-user",
                approved_answer=reviewed_answer,
            )

        self.assertTrue(approved)
        self.assertEqual(save_correction.call_args.args[1], "Who founded the school?")
        self.assertEqual(save_correction.call_args.args[2], reviewed_answer)
        self.assertEqual(save_correction.call_args.kwargs["created_by"], "staff-user")
        self.assertTrue(update.committed)


if __name__ == "__main__":
    unittest.main()
