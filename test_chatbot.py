import requests
import time
import json
from datetime import datetime

# ====================== CONFIG ======================
CHATBOT_URL = "http://127.0.0.1:5000/chat"          # ← change if needed
SESSION_ID = f"mixed_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DELAY_BETWEEN_QUESTIONS = 1.3
# ====================================================

# Grouped so the report is easy to read
TEST_CASES = [
    # ========== 1. Handbook – Identity / History ==========
    {"cat": "Handbook-Identity", "q": "What's the real full name of this school and where are the campuses?"},
    {"cat": "Handbook-Identity", "q": "What was PAC called before it became a college?"},
    {"cat": "Handbook-Identity", "q": "When did this school start and who founded it?"},
    {"cat": "Handbook-Identity", "q": "What does the school logo mean?"},
    {"cat": "Handbook-Identity", "q": "Who wrote the PAC hymn?"},
    {"cat": "Handbook-Identity", "q": "Something about 1975?"},
    {"cat": "Handbook-Identity", "q": "Is there anything about the Macias family?"},

    # ========== 2. Handbook – Vision / Mission / Values ==========
    {"cat": "Handbook-Vision", "q": "What's the school's philosophy about education?"},
    {"cat": "Handbook-Vision", "q": "What does PAC want to become in the future?"},
    {"cat": "Handbook-Vision", "q": "What are the core values again? The PAC ones."},
    {"cat": "Handbook-Vision", "q": "Is there something about Christ-centered living?"},
    {"cat": "Handbook-Vision", "q": "Any mention of Ellen White or Education book?"},

    # ========== 3. Handbook – Demerit / Conduct ==========
    {"cat": "Handbook-Conduct", "q": "How many points do we start with each semester?"},
    {"cat": "Handbook-Conduct", "q": "What's a first degree offense worth in demerits?"},
    {"cat": "Handbook-Conduct", "q": "What's the heaviest demerit level?"},
    {"cat": "Handbook-Conduct", "q": "What happens if you get 20 or more demerits?"},
    {"cat": "Handbook-Conduct", "q": "How long do I have to appeal a disciplinary decision?"},
    {"cat": "Handbook-Conduct", "q": "Can we wear jewelry like rings or necklaces?"},
    {"cat": "Handbook-Conduct", "q": "What's the dress code for students?"},
    {"cat": "Handbook-Conduct", "q": "Are slippers allowed on campus?"},
    {"cat": "Handbook-Conduct", "q": "Is public display of affection okay?"},
    {"cat": "Handbook-Conduct", "q": "What happens if you miss worship too many times?"},

    # ========== 4. Handbook – Legal / Policies ==========
    {"cat": "Handbook-Legal", "q": "Is there an anti-bullying law in the handbook?"},
    {"cat": "Handbook-Legal", "q": "What's the Safe Spaces Act and its RA number?"},
    {"cat": "Handbook-Legal", "q": "What about RA 11313?"},
    {"cat": "Handbook-Legal", "q": "Is hazing allowed or mentioned?"},

    # ========== 5. Programs / FAQ-style ==========
    {"cat": "Programs-FAQ", "q": "What programs or courses does PAC actually offer?"},
    {"cat": "Programs-FAQ", "q": "What senior high strands are available?"},
    {"cat": "Programs-FAQ", "q": "Is there Nursing or Midwifery?"},
    {"cat": "Programs-FAQ", "q": "Any TESDA courses?"},
    {"cat": "Programs-FAQ", "q": "Do they have Education programs?"},
    {"cat": "Programs-FAQ", "q": "Is Business Administration available?"},
    {"cat": "Programs-FAQ", "q": "Are there scholarships for siblings?"},
    {"cat": "Programs-FAQ", "q": "What about dormitory fees or charges?"},
    {"cat": "Programs-FAQ", "q": "Something about drug test for first years?"},

    # ========== 6. Date / Schedule questions ==========
    {"cat": "Schedule", "q": "Do we have classes this week?"},
    {"cat": "Schedule", "q": "Is there class tomorrow?"},
    {"cat": "Schedule", "q": "Are there any holidays this month?"},
    {"cat": "Schedule", "q": "Does high school have classes this week?"},
    {"cat": "Schedule", "q": "Is the office open on Friday?"},
    {"cat": "Schedule", "q": "Any suspension announced?"},

    # ========== 7. Short Follow-ups (critical) ==========
    # These must be asked AFTER a schedule question in the same session
    {"cat": "Follow-up", "q": "so all level?"},
    {"cat": "Follow-up", "q": "college too?"},
    {"cat": "Follow-up", "q": "elementary also?"},
    {"cat": "Follow-up", "q": "sa tanan?"},
    {"cat": "Follow-up", "q": "for everyone?"},
    {"cat": "Follow-up", "q": "whole school?"},



    # ========== 9. Vague / Elliptical ==========
    {"cat": "Vague", "q": "Something about the 2025 edition?"},
    {"cat": "Vague", "q": "Any updates for school year 2025-2026?"},
    {"cat": "Vague", "q": "What about the Week of Spiritual Emphasis?"},
    {"cat": "Vague", "q": "Is there a deadline for incomplete grades?"},
    {"cat": "Vague", "q": "Any policy from MORPHE of 2008?"},
    {"cat": "Vague", "q": "What's the residency limit for finishing a degree?"},
]

# Phrases that usually mean the bot failed / fell back / only disambiguated
FALLBACK_PHRASES = [
    "i don't have a confirmed answer",
    "i've saved it so our staff",
    "feel free to ask me something else",
    "couldn't quite understand",
    "could you try rephrasing",
    "i can only answer pac-related questions",
    "testing",
    "cannot_answer",
    "i'm not sure",
    "no information",
    "i found a few handbook sections",
    "which one are you asking about",
    "which one did you mean",
    "i don't know",
    "sorry, i can't",
    "i don't have that information",
]


def is_bad_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 20:
        return True
    lower = answer.lower()
    if any(phrase in lower for phrase in FALLBACK_PHRASES):
        return True
    return False


def ask_chatbot(question: str) -> str:
    payload = {
        "message": question,
        "session_id": SESSION_ID,
    }
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    try:
        response = requests.post(
            CHATBOT_URL,
            json=payload,
            headers=headers,
            timeout=50,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return (
                data.get("reply")
                or data.get("answer")
                or data.get("response")
                or data.get("message")
                or str(data)
            )
        return str(data)
    except Exception as e:
        return f"[ERROR] {str(e)}"


def main():
    results = []
    print(f"\nStarting MIXED test suite — {len(TEST_CASES)} questions")
    print(f"Session ID: {SESSION_ID}")
    print("=" * 80)

    current_cat = None
    for i, item in enumerate(TEST_CASES, 1):
        cat = item["cat"]
        question = item["q"]

        if cat != current_cat:
            print(f"\n{'─'*30} {cat} {'─'*30}")
            current_cat = cat

        print(f"\n[{i:02d}/{len(TEST_CASES)}] {question}")
        answer = ask_chatbot(question)
        failed = is_bad_answer(answer)
        status = "❌ FAILED" if failed else "✅ PASSED"
        print(f" → {status}")
        print(f" → {answer[:220]}{'...' if len(answer) > 220 else ''}")

        results.append({
            "number": i,
            "category": cat,
            "question": question,
            "answer": answer,
            "status": "failed" if failed else "passed",
            "timestamp": datetime.now().isoformat(),
        })
        time.sleep(DELAY_BETWEEN_QUESTIONS)

    # ===== Summary by category =====
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["status"] == "passed":
            by_cat[r["category"]]["passed"] += 1

    answered = sum(1 for r in results if r["status"] == "passed")

    # Save JSON
    with open("test_results_mixed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save Markdown report
    with open("test_results_mixed.md", "w", encoding="utf-8") as f:
        f.write("# Chatbot Mixed Test Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Session:** {SESSION_ID}\n\n")
        f.write(f"**Overall Score:** {answered}/{len(results)} passed\n\n")

        f.write("## Score by Category\n\n")
        for cat, stats in by_cat.items():
            f.write(f"- **{cat}**: {stats['passed']}/{stats['total']}\n")
        f.write("\n---\n\n")

        for r in results:
            icon = "✅" if r["status"] == "passed" else "❌"
            f.write(f"### {icon} Q{r['number']} [{r['category']}]: {r['question']}\n\n")
            f.write(f"**Answer:**\n\n{r['answer']}\n\n---\n\n")

    print("\n" + "=" * 80)
    print("Finished!")
    print("Results saved to: test_results_mixed.json & test_results_mixed.md")
    print(f"\nOverall Score: {answered}/{len(results)} passed")
    print("\nBy category:")
    for cat, stats in by_cat.items():
        print(f"  {cat:20} {stats['passed']:2}/{stats['total']}")


if __name__ == "__main__":
    main()