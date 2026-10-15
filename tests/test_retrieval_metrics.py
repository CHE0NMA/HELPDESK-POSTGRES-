import unittest
from unittest.mock import patch

from ai_providers.metrics import ModelMetrics
from services.embeddings import get_embedding_coverage


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, query):
        self.query = query

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return FakeCursor(self.rows)

    def close(self):
        pass


class EmbeddingCoverageTests(unittest.TestCase):
    def test_coverage_counts_and_readiness_are_explicit(self):
        rows = [("faq", 10, 8), ("handbook_chunks", 4, 0)]
        with patch("services.embeddings.is_configured", return_value=True):
            result = get_embedding_coverage(lambda: FakeConnection(rows))

        self.assertEqual(result["faq"]["coverage_percent"], 80.0)
        self.assertTrue(result["faq"]["semantic_ready"])
        self.assertEqual(result["handbook_chunks"]["coverage_percent"], 0.0)
        self.assertFalse(result["handbook_chunks"]["semantic_ready"])

    def test_unconfigured_provider_is_not_reported_semantic_ready(self):
        rows = [("faq", 3, 3), ("handbook_chunks", 0, 0)]
        with patch("services.embeddings.is_configured", return_value=False):
            result = get_embedding_coverage(lambda: FakeConnection(rows))

        self.assertFalse(result["faq"]["semantic_ready"])
        self.assertEqual(result["faq"]["coverage_percent"], 100.0)
        self.assertEqual(result["handbook_chunks"]["coverage_percent"], 0.0)


class ProviderMetricsTests(unittest.TestCase):
    def test_provider_outcomes_are_counted_separately(self):
        metrics = ModelMetrics()
        metrics.record_response_time(12, True, provider="groq")
        metrics.record_response_time(24, False, provider="gemini")

        providers = metrics.get_provider_stats()
        self.assertNotIn("providers", metrics.get_stats())
        self.assertEqual(providers["groq"]["successful_calls"], 1)
        self.assertEqual(providers["groq"]["failed_calls"], 0)
        self.assertEqual(providers["gemini"]["successful_calls"], 0)
        self.assertEqual(providers["gemini"]["failed_calls"], 1)


if __name__ == "__main__":
    unittest.main()