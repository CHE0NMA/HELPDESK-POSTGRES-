import unittest

from services.answer_engine import get_top_faq_candidates


class FaqMetadataRankingTests(unittest.TestCase):
    def test_related_questions_match_pipe_separated_alternates(self):
        faq_row = (
            "Submit a transcript request to the Registrar.",
            "How do I request academic records?",
            1,
            "Registrar",
            "transcript_request",
            "transcript, records",
            "How can I request my TOR?|Where do I get my transcript?",
            0,
            None,
            None,
        )

        candidates = get_top_faq_candidates(
            "Where can I get my TOR?",
            "where can i get my transcript of records",
            [faq_row],
        )

        self.assertEqual(candidates[0]["faq"], faq_row)
        self.assertGreaterEqual(candidates[0]["related_question_score"], 0.60)

    def test_keywords_and_topic_intent_contribute_to_candidate_ranking(self):
        question = "Who handles TES subsidy claims?"
        expanded = "who handles tertiary education subsidy claims"
        with_metadata = (
            "Submit a TES claim to the Student Finance Office.",
            "How do I apply for the subsidy?",
            1,
            "Student Finance",
            "TES_claim",
            "tertiary education subsidy, claim",
            "",
            0,
            None,
            None,
        )
        without_metadata = (
            with_metadata[0], with_metadata[1], 2,
            "", "", "", "", 0, None, None,
        )

        ranked = get_top_faq_candidates(
            question,
            expanded,
            [without_metadata, with_metadata],
        )
        by_id = {candidate["faq"][2]: candidate for candidate in ranked}

        self.assertGreater(
            by_id[1]["keyword_coverage"],
            by_id[2]["keyword_coverage"],
        )
        self.assertGreater(
            by_id[1]["topic_intent_coverage"],
            by_id[2]["topic_intent_coverage"],
        )
        self.assertGreater(by_id[1]["combined"], by_id[2]["combined"])


if __name__ == "__main__":
    unittest.main()
