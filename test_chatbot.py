import requests
import time
import json
from datetime import datetime

# ====================== CONFIG ======================
CHATBOT_URL = "http://127.0.0.1:5000/chat"          # ← change if needed
# Unique every run so conversation context + any session-tied cache can't stick
SESSION_ID = f"handbook_only_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DELAY_BETWEEN_QUESTIONS = 1.5
# ====================================================

# Only handbook / identity / discipline / legal questions
QUESTIONS = [
    # ========== Identity / History ==========
    "What's the real full name of this school and where are the campuses?",
    "What was PAC called before it became a college?",
    "When did this school start and who founded it?",
    "What does the school logo mean?",
    "Who wrote the PAC hymn?",

    # ========== Philosophy / Vision / Mission / Values ==========
    "What's the school's philosophy about education?",
    "What does PAC want to become in the future?",
    "What's the main mission of the college?",
    "What are the core values again? The PAC ones.",
    "What is the vision of PAC in simple words?",
    "Tell me about the school philosophy again.",

    # ========== Merit / Demerit system ==========
    "How many points do we start with each semester?",
    "What's a first degree offense worth in demerits?",
    "How bad is a second-degree offense?",
    "What's the heaviest demerit level?",
    "Give examples of light or first-degree offenses.",
    "What counts as a serious or third-degree offense?",
    "What happens if you get 20 or more demerits?",
    "Is there a maximum fine for violations?",
    "How long do I have to appeal a disciplinary decision?",

    # ========== Dress / Conduct / Campus rules ==========
    "Can we wear jewelry like rings or necklaces?",
    "What's the dress code for students?",
    "Is it okay to go on hikes without a chaperone?",
    "Are we allowed to visit other dorms freely?",
    "Can we cook inside the dorm?",
    "What food is not allowed in the cafeteria?",

    # ========== Legal / Policies ==========
    "Is there an anti-bullying law in the handbook?",
    "What's the Safe Spaces Act and its RA number?",

    # ========== Programs ==========
    "What programs or courses does PAC actually offer?",
    "What senior high strands are available?",
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
]


def is_bad_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 15:
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
            timeout=45,
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
    print(f"\nStarting handbook-only test — {len(QUESTIONS)} questions")
    print(f"Session ID: {SESSION_ID}")
    print("=" * 75)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(QUESTIONS)}] {question}")
        answer = ask_chatbot(question)
        failed = is_bad_answer(answer)
        status = "❌ FAILED" if failed else "✅ PASSED"
        print(f" → {status}")
        print(f" → {answer[:200]}{'...' if len(answer) > 200 else ''}")

        results.append({
            "number": i,
            "question": question,
            "answer": answer,
            "status": "failed" if failed else "passed",
            "timestamp": datetime.now().isoformat(),
        })
        time.sleep(DELAY_BETWEEN_QUESTIONS)

    # ===== Save results =====
    answered = sum(1 for r in results if r["status"] == "passed")
    with open("test_results_handbook.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open("test_results_handbook.md", "w", encoding="utf-8") as f:
        f.write("# Chatbot Handbook-Only Test Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Session:** {SESSION_ID}\n\n")
        f.write(f"**Score:** {answered}/{len(results)} passed\n\n")
        f.write("---\n\n")
        for r in results:
            icon = "✅" if r["status"] == "passed" else "❌"
            f.write(f"### {icon} Q{r['number']}: {r['question']}\n\n")
            f.write(f"**Answer:**\n\n{r['answer']}\n\n---\n\n")

    print("\n" + "=" * 75)
    print("Finished!")
    print("Results saved to: test_results_handbook.json & test_results_handbook.md")
    print(f"\nFinal Score: {answered}/{len(results)} passed")


if __name__ == "__main__":
    main()