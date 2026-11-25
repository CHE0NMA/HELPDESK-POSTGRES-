import unittest
from unittest.mock import Mock, patch

from services import hybrid_search
from services.embeddings import _store_embedding_row


class HybridSearchTests(unittest.TestCase):
    def test_pgvector_is_disabled_without_opt_in(self):
        get_db = Mock(side_effect=AssertionError("database should not be queried"))
        with patch.dict("os.environ", {"PGVECTOR_HYBRID_SEARCH": "0"}):
            status = hybrid_search.get_hybrid_search_status(get_db)

        self.assertEqual(status, {"enabled": False, "available": False})
        get_db.assert_not_called()

    def test_only_default_gemini_vector_dimensions_use_pgvector(self):
        self.assertIsNone(hybrid_search._embedding_literal([0.1, 0.2]))
        vector = hybrid_search._embedding_literal([0.1] * 3072)
        self.assertTrue(vector.startswith("["))
        self.assertTrue(vector.endswith("]"))

    def test_faq_search_unions_all_retrieval_candidates(self):
        cursor = Mock()
        cursor.fetchone.return_value = (True, True, True)
        cursor.fetchall.return_value = [(3,), (8,), (13,)]
        connection = Mock()
        connection.cursor.return_value = cursor

        with patch.dict("os.environ", {"PGVECTOR_HYBRID_SEARCH": "1"}):
            candidates = hybrid_search.search_faq_candidate_ids(
                lambda: connection,
                [0.1] * 3072,
                ["unsaon pag-enroll", "how do I enroll"],
            )

        self.assertEqual(candidates, {3, 8, 13})
        self.assertIn("vector_hits", cursor.execute.call_args_list[-1].args[0])
        self.assertIn("fts_hits", cursor.execute.call_args_list[-1].args[0])
        self.assertIn("trigram_hits", cursor.execute.call_args_list[-1].args[0])
        connection.close.assert_called_once()

    def test_search_falls_back_when_indexes_are_not_installed(self):
        cursor = Mock()
        cursor.fetchone.return_value = (False, False, False)
        connection = Mock()
        connection.cursor.return_value = cursor

        with patch.dict("os.environ", {"PGVECTOR_HYBRID_SEARCH": "1"}):
            result = hybrid_search.search_faq_candidate_ids(
                lambda: connection,
                [0.1] * 3072,
                ["How do I enroll?"],
            )

        self.assertIsNone(result)
        self.assertEqual(cursor.execute.call_count, 1)
        connection.close.assert_called_once()

    def test_status_reports_enabled_index_coverage(self):
        cursor = Mock()
        cursor.fetchone.return_value = ("0.8.0", True, True, True, False)
        connection = Mock()
        connection.cursor.return_value = cursor

        with patch.dict("os.environ", {"PGVECTOR_HYBRID_SEARCH": "1"}):
            status = hybrid_search.get_hybrid_search_status(lambda: connection)

        self.assertTrue(status["enabled"])
        self.assertFalse(status["available"])
        self.assertTrue(status["indexes"]["faq_vector"])
        self.assertFalse(status["indexes"]["handbook_vector"])

    def test_embedding_write_updates_jsonb_and_halfvec_when_available(self):
        cursor = Mock()
        cursor.fetchone.return_value = (True,)

        _store_embedding_row(cursor, "faq", "id", 12, [0.25] * 3072)

        self.assertIn("embedding_vector = %s::halfvec(3072)", cursor.execute.call_args.args[0])
        self.assertEqual(cursor.execute.call_args.args[1][-1], 12)


if __name__ == "__main__":
    unittest.main()
