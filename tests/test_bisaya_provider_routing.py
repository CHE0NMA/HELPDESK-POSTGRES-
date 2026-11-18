import unittest
from unittest.mock import patch

from ai_providers import providers
from services import faq, language
from services.conversation import ConversationContext


class BisayaProviderRoutingTests(unittest.TestCase):
    def test_bisaya_request_uses_gemini_only_for_main_and_fast_calls(self):
        calls = []

        def fake_groq(prompt, max_tokens):
            calls.append("groq")
            return "groq response"

        def fake_gemini(prompt, max_tokens):
            calls.append("gemini")
            return "gemini response"

        with patch.dict(
            providers._PROVIDER_READY,
            {"groq": lambda: True, "gemini": lambda: True},
        ), patch.dict(
            providers._PROVIDER_FUNCS,
            {"groq": fake_groq, "gemini": fake_gemini},
        ), patch.dict(
            providers._FAST_PROVIDER_FUNCS,
            {"groq": fake_groq, "gemini": fake_gemini},
        ), patch.object(providers.circuit_breaker, "is_open", return_value=False), \
             patch.object(providers.circuit_breaker, "record_success"), \
             patch.object(providers.model_metrics, "record_response_time"):
            with providers.provider_for_language("bisaya"):
                self.assertEqual(providers.call_with_fallback("prompt"), "gemini response")
                self.assertEqual(
                    providers.call_fast_with_fallback("prompt"), "gemini response"
                )

        self.assertEqual(calls, ["gemini", "gemini"])

    def test_bisaya_provider_failure_does_not_fall_back_to_groq(self):
        calls = []

        def fake_groq(prompt, max_tokens):
            calls.append("groq")
            return "groq response"

        def failed_gemini(prompt, max_tokens):
            calls.append("gemini")
            return ""

        with patch.dict(
            providers._PROVIDER_READY,
            {"groq": lambda: True, "gemini": lambda: True},
        ), patch.dict(
            providers._PROVIDER_FUNCS,
            {"groq": fake_groq, "gemini": failed_gemini},
        ), patch.object(providers.circuit_breaker, "is_open", return_value=False), \
             patch.object(providers.circuit_breaker, "record_failure"), \
             patch.object(providers.model_metrics, "record_response_time"):
            with providers.provider_for_language("bisaya"):
                result = providers.call_with_fallback("prompt")

        self.assertEqual(result, "")
        self.assertEqual(calls, ["gemini"])

    def test_faq_request_pins_bisaya_but_not_tagalog_to_gemini(self):
        context = ConversationContext("provider-routing-test")

        def report_provider(question, return_topic, session_id):
            return providers._PROVIDER_OVERRIDE.get()

        with patch.object(faq, "get_conversation_context", return_value=context), \
             patch.object(faq, "_find_answer", side_effect=report_provider):
            self.assertEqual(faq.find_answer("kinsa ang dean sa computer studies?"), "gemini")
            self.assertIsNone(faq.find_answer("sino ang dean ng computer studies?"))

    def test_parallel_faq_scoring_inherits_gemini_provider_pin(self):
        def handbook_score(question, query_embedding, detected_lang):
            return (
                1.0 if providers._PROVIDER_OVERRIDE.get() == "gemini" else 0.0,
                [],
            )

        def faq_score(question, detected_lang, query_embedding=None):
            score = 1.0 if providers._PROVIDER_OVERRIDE.get() == "gemini" else 0.0
            return [], score

        with patch.object(faq, "_handbook_best_score", side_effect=handbook_score), \
             patch.object(faq, "_faq_candidates_and_score", side_effect=faq_score), \
             providers.provider_for_language("bisaya"):
            result = faq._score_sources_parallel("kinsa", "bisaya")

        self.assertEqual(result["hb_score"], 1.0)
        self.assertEqual(result["faq_score"], 1.0)

    def test_ambiguous_language_detection_uses_gemini_fast_path(self):
        with patch.dict(language._LANG_CACHE, {}, clear=True), \
             patch.object(language.circuit_breaker, "is_open", return_value=False), \
             patch.object(
                 language, "llm_chat_fast_gemini", return_value="bisaya"
             ) as gemini, \
             patch.object(language, "llm_chat_fast") as groq:
            result = language.detect_language("wala")

        self.assertEqual(result, "bisaya")
        gemini.assert_called_once()
        groq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
