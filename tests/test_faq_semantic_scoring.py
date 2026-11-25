import unittest
from unittest.mock import Mock, patch

from services.answer_engine import get_top_faq_candidates
from services import embeddings, faq
from services.embeddings import embedding_text_for_faq, embedding_text_for_chunk


class FAQSemanticScoringTests(unittest.TestCase):
    def setUp(self):
        self.faq = (
            "Late enrollment is handled by the admissions office.",
            "How do I enroll late?",
            1,
            "Enrollment",
            "Late enrollment",
            "admissions",
            "Can I still enroll after the deadline?",
            0,
            None,
            [1.0, 0.0],
        )

    def test_strong_semantic_match_can_win_without_shared_words(self):
        candidates = get_top_faq_candidates(
            "unsaon ko pagpadayon bisan nalapas na",
            "unsaon ko pagpadayon bisan nalapas na",
            [self.faq],
            detected_lang="bisaya",
            query_embedding=[1.0, 0.0],
        )

        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0]["fuzzy"], 0.60)
        self.assertGreaterEqual(candidates[0]["combined"], 0.99)

    def test_bisaya_faq_lookup_embeds_the_original_question(self):
        question = "unsaon ko pagpadayon bisan nalapas na"
        with patch.object(faq, "_get_query_embedding", return_value=[1.0, 0.0]) as embed, \
             patch.object(faq.lang_service, "normalize_vernacular_spelling", side_effect=lambda text: text), \
             patch.object(faq.lang_service, "expand_query", side_effect=lambda text: text), \
             patch.object(faq, "llm_expand_faq_query", return_value=[]):
            candidates = faq._get_faq_candidates(question, "bisaya", [self.faq])

        embed.assert_called_once_with(question)
        self.assertEqual(candidates[0]["faq"][2], 1)

    def test_unembedded_candidate_keeps_its_lexical_score(self):
        faq_without_embedding = self.faq[:9]
        question = "How do I enroll late?"
        lexical = get_top_faq_candidates(
            question, question, [faq_without_embedding]
        )[0]
        with_embedding_query = get_top_faq_candidates(
            question, question, [faq_without_embedding],
            query_embedding=[0.0, 1.0],
        )[0]

        self.assertEqual(with_embedding_query["combined"], lexical["combined"])

    def test_faq_embedding_includes_aliases_and_answer_context(self):
        text = embedding_text_for_faq(
            "How do I enroll late?",
            "admissions",
            "unsaon pag-enroll kung late na ko",
            "Contact admissions for late enrollment.",
            "Enrollment",
            "Late enrollment",
        )

        self.assertIn("unsaon pag-enroll kung late na ko", text)
        self.assertIn("Contact admissions", text)
        self.assertIn("Enrollment", text)

    def test_faq_embedding_keeps_answer_topics_past_first_500_characters(self):
        answer = ("Background details. " * 35) + "Unique policy requirement."
        text = embedding_text_for_faq(
            "What is the policy?",
            answer=answer,
        )

        self.assertIn("Unique policy requirement.", text)

    def test_handbook_embedding_includes_topic_tags_and_key_facts(self):
        text = embedding_text_for_chunk(
            "Enrollment",
            "Students submit the required forms.",
            topic_tags=["admissions", "late enrollment"],
            key_facts=["Late applications go to the admissions office."],
        )

        self.assertIn("late enrollment", text)
        self.assertIn("admissions office", text)

    def test_embedding_request_uses_query_and_document_task_types(self):
        response = Mock(status_code=200, text="")
        response.json.return_value = {"embedding": {"values": [0.1, 0.2]}}
        with patch.object(embeddings, "GEMINI_API_KEYS", ["test-key"]), \
             patch.object(embeddings, "get_next_gemini_key", return_value="test-key"), \
             patch.object(embeddings.requests, "post", return_value=response) as post, \
             patch.object(embeddings, "mark_gemini_success"):
            embeddings.get_embedding("question", task_type="RETRIEVAL_QUERY")
            embeddings.get_embedding("faq document", task_type="RETRIEVAL_DOCUMENT")

        self.assertEqual(
            [call.kwargs["json"]["taskType"] for call in post.call_args_list],
            ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"],
        )


if __name__ == "__main__":
    unittest.main()
