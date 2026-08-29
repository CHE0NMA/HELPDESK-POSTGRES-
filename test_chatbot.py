import requests
import time
import json
from datetime import datetime
from collections import defaultdict

# ====================== CONFIG ======================
CHATBOT_URL = "http://127.0.0.1:5000/chat"
SESSION_ID = f"mixed_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DELAY_BETWEEN_QUESTIONS = 1.4
# ====================================================

TEST_CASES = [
    # ========== 1. Identity / History ==========
    {"cat": "Handbook-Identity", "q": "What is the full official name of the school and its campuses?"},
    {"cat": "Handbook-Identity", "q": "What was PAC originally called before it became a college?"},
    {"cat": "Handbook-Identity", "q": "When was Philippine Advent College founded and who were the founders?"},
    {"cat": "Handbook-Identity", "q": "What does the official PAC logo symbolize?"},
    {"cat": "Handbook-Identity", "q": "Who composed the music and wrote the lyrics of the PAC Hymn?"},
    {"cat": "Handbook-Identity", "q": "What is the significance of the year 1975 in PAC history?"},
    {"cat": "Handbook-Identity", "q": "What role did the Macias family play in founding PAC?"},

    # ========== 2. Vision / Mission / Values ==========
    {"cat": "Handbook-Vision", "q": "What is the philosophy of education at Philippine Advent College?"},
    {"cat": "Handbook-Vision", "q": "What is PAC’s vision statement?"},
    {"cat": "Handbook-Vision", "q": "What are the three core values of PAC and what do they mean?"},
    {"cat": "Handbook-Vision", "q": "How does PAC describe being Christ-centered?"},
    {"cat": "Handbook-Vision", "q": "Is there a quotation from Ellen G. White’s book Education in the student manual?"},

    # ========== 3. Demerit / Conduct ==========
    {"cat": "Handbook-Conduct", "q": "How many merit points does every student start with each semester?"},
    {"cat": "Handbook-Conduct", "q": "How many demerits is a first-degree offense?"},
    {"cat": "Handbook-Conduct", "q": "How many demerits is a third-degree offense?"},
    {"cat": "Handbook-Conduct", "q": "What happens when a student reaches 20 or more demerits?"},
    {"cat": "Handbook-Conduct", "q": "How many days does a student have to appeal a decision of the Student Disciplinary Committee?"},
    {"cat": "Handbook-Conduct", "q": "Is jewelry such as rings, bracelets, or necklaces allowed on campus?"},
    {"cat": "Handbook-Conduct", "q": "What is the official dress and appearance policy for students?"},
    {"cat": "Handbook-Conduct", "q": "Are slippers allowed on campus?"},
    {"cat": "Handbook-Conduct", "q": "Is public display of affection considered a violation?"},
    {"cat": "Handbook-Conduct", "q": "What is the policy on missing dormitory worships?"},

    # ========== 4. Legal ==========
    {"cat": "Handbook-Legal", "q": "Does the student manual mention the Anti-Bullying Act or RA 10627?"},
    {"cat": "Handbook-Legal", "q": "What is the Safe Spaces Act and what is its Republic Act number?"},
    {"cat": "Handbook-Legal", "q": "What does RA 11313 cover according to the handbook?"},
    {"cat": "Handbook-Legal", "q": "Is hazing allowed or prohibited in PAC?"},

    # ========== 5. Programs / Fees ==========
    {"cat": "Programs-FAQ", "q": "What degree programs does Philippine Advent College offer?"},
    {"cat": "Programs-FAQ", "q": "What senior high school strands are available at PAC?"},
    {"cat": "Programs-FAQ", "q": "Does PAC offer Bachelor of Science in Nursing and Midwifery?"},
    {"cat": "Programs-FAQ", "q": "What TESDA NC II programs are offered?"},
    {"cat": "Programs-FAQ", "q": "What teacher education programs are available?"},
    {"cat": "Programs-FAQ", "q": "Is Bachelor of Science in Business Administration offered?"},
    {"cat": "Programs-FAQ", "q": "Is there a family scholarship or discount for siblings?"},
    {"cat": "Programs-FAQ", "q": "What are the current dormitory charges?"},
    {"cat": "Programs-FAQ", "q": "Is a drug test required for incoming first-year students?"},

    # ========== 6. Schedule + immediate follow-ups (critical) ==========
    # Each block is one schedule question followed by its real clarifying follow-ups
    {"cat": "Schedule", "q": "Do we have classes this week?"},
    {"cat": "Follow-up", "q": "Does that apply to all levels?"},
    {"cat": "Follow-up", "q": "College too?"},
    {"cat": "Follow-up", "q": "Elementary also?"},

    {"cat": "Schedule", "q": "Is there class tomorrow?"},
    {"cat": "Follow-up", "q": "For the whole school?"},
    {"cat": "Follow-up", "q": "sa tanan?"},

    {"cat": "Schedule", "q": "Are there any holidays this month?"},
    {"cat": "Follow-up", "q": "Does high school also have no classes?"},
    {"cat": "Follow-up", "q": "for everyone?"},

    {"cat": "Schedule", "q": "Does high school have classes this week?"},
    {"cat": "Follow-up", "q": "What about college?"},
    {"cat": "Follow-up", "q": "elementary also?"},

    {"cat": "Schedule", "q": "Is the office open on Friday?"},
    {"cat": "Follow-up", "q": "All offices or just Student Affairs?"},

    {"cat": "Schedule", "q": "Any suspension announced?"},
    {"cat": "Follow-up", "q": "Is it for all campuses?"},
    {"cat": "Follow-up", "q": "whole school?"},

    # ========== 7. Vague ==========
    {"cat": "Vague", "q": "Something about the 2025 edition of the student manual?"},
    {"cat": "Vague", "q": "Any updates for school year 2025-2026?"},
    {"cat": "Vague", "q": "What about the Week of Spiritual Emphasis?"},
    {"cat": "Vague", "q": "Is there a deadline for completing incomplete grades?"},
    {"cat": "Vague", "q": "Any policy from MORPHE of 2008?"},
    {"cat": "Vague", "q": "What is the maximum residency period to finish a degree?"},
]

# Stronger failure detection
FALLBACK_PHRASES = [
    "i don't have a confirmed answer",
    "i've saved it so our staff",
    "feel free to ask me something else",
    "couldn't quite understand",
    "could you try rephrasing",
    "i can only answer pac-related questions",
    "cannot_answer",
    "i'm not sure",
    "no information",
    "i found a few handbook sections",
    "which one are you asking about",
    "which one did you mean",
    "i don't know",
    "sorry, i can't",
    "i don't have that information",
    "i don't have specific information",
]

# Phrases that almost always mean the wrong chunk was retrieved
WRONG_CHUNK_MARKERS = [
    "the treasurer shall only be the one",
    "regular meetings of the members",
    "last friday of the month around 1:30",
    "learning management system",
    "internship fee(for education",
    "office work -------------------------- php",
    "field work -------------------------- php",
    "enrollee for 1st year",
    "who graduated with highest honor",
    "refrain from any form of harassment",
    "gender-based sexual harassment committee",
    "sport development program of philippine advent",
    "mga katungdanan sa auditor",
    "liquidation of all the expenses",
]


def is_bad_answer(answer: str, question: str = "") -> bool:
    if not answer or len(answer.strip()) < 30:
        return True
    lower = answer.lower()
    if any(p in lower for p in FALLBACK_PHRASES):
        return True
    if any(m in lower for m in WRONG_CHUNK_MARKERS):
        return True
    # Extra hard checks for known high-value questions
    q = question.lower()
    if "jewelry" in q or "necklace" in q or "ring" in q:
        if "wedding ring" not in lower and "not permitted" not in lower and "jewelry" not in lower:
            return True
    if "slipper" in q:
        if "slipper" not in lower and "inappropriate attire" not in lower:
            return True
    if "worship" in q and "miss" in q:
        if "worship" not in lower and "absence" not in lower:
            return True
    if "ellen" in q or "white" in q:
        if "ellen" not in lower and "white" not in lower and "education" not in lower:
            return True
    if "anti-bullying" in q or "10627" in q:
        if "10627" not in lower and "bullying" not in lower:
            return True
    if "residency" in q or "maximum" in q and "year" in q:
        if "5 year" not in lower and "five year" not in lower and "maximum of 5" not in lower:
            return True
    return False


def ask_chatbot(question: str) -> str:
    payload = {"message": question, "session_id": SESSION_ID}
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    try:
        r = requests.post(CHATBOT_URL, json=payload, headers=headers, timeout=55)
        r.raise_for_status()
        data = r.json()
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
        return f"[ERROR] {e}"


def main():
    results = []
    print(f"\nStarting MIXED test — {len(TEST_CASES)} questions")
    print(f"Session: {SESSION_ID}")
    print("=" * 80)

    current_cat = None
    for i, item in enumerate(TEST_CASES, 1):
        cat = item["cat"]
        q = item["q"]
        if cat != current_cat:
            print(f"\n{'─'*28} {cat} {'─'*28}")
            current_cat = cat

        print(f"\n[{i:02d}/{len(TEST_CASES)}] {q}")
        answer = ask_chatbot(q)
        failed = is_bad_answer(answer, q)
        status = "❌ FAILED" if failed else "✅ PASSED"
        print(f" → {status}")
        print(f" → {answer[:240]}{'...' if len(answer) > 240 else ''}")

        results.append({
            "number": i,
            "category": cat,
            "question": q,
            "answer": answer,
            "status": "failed" if failed else "passed",
            "timestamp": datetime.now().isoformat(),
        })
        time.sleep(DELAY_BETWEEN_QUESTIONS)

    by_cat = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["status"] == "passed":
            by_cat[r["category"]]["passed"] += 1

    answered = sum(1 for r in results if r["status"] == "passed")

    with open("test_results_mixed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
    print(f"Finished! Overall: {answered}/{len(results)} passed")
    print("By category:")
    for cat, stats in by_cat.items():
        print(f"  {cat:20} {stats['passed']:2}/{stats['total']}")


if __name__ == "__main__":
    main()