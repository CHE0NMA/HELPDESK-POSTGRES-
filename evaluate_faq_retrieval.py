"""Run Recall@k and MRR against a JSON file of reviewed FAQ retrieval cases.

Each case should contain `question`, `expected_faq_id` (or
`expected_faq_ids`), and optionally `language`.
"""
import argparse
import json
from pathlib import Path

from services import faq
from services.retrieval_evaluation import evaluate_retrieval_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON file containing reviewed cases")
    args = parser.parse_args()

    with args.cases.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("The case file must contain a JSON array")

    rows = faq._get_all_faqs()
    report = evaluate_retrieval_cases(
        cases,
        lambda case: [
            candidate["faq"][2]
            for candidate in faq._get_faq_candidates(
                case["question"],
                case.get("language", "english"),
                rows,
            )
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
