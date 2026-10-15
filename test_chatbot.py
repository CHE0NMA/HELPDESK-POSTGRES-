#!/usr/bin/env python3
"""
test_chatbot.py — mixed regression suite for the PAC helpdesk bot.

Usage:
    python test_chatbot.py

Requires the Flask app running on CHATBOT_URL (default http://127.0.0.1:5000/chat).
"""

import requests
import time
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import Any

# ====================== CONFIG ======================
CHATBOT_URL = "http://127.0.0.1:5001/chat"
SESSION_ID = f"mixed_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DELAY_BETWEEN_QUESTIONS = 2.0          # was 1.4 — give LLM + DB pool room
REQUEST_TIMEOUT = 90                   # was 55
DEBUG_FIRST_N = 3                      # print full status/body for first N calls
# ====================================================

TEST_CASES: list[dict[str, Any]] = [
    # ========== 1. Identity / History ==========
    {"cat": "Handbook-Identity", "q": "What is the full official name of the school and its campuses?"},
    {"cat": "Handbook-Identity", "q": "What was PAC originally called before it became a college?", "must_contain": ["hillside view", "elementary school"]},
    {"cat": "Handbook-Identity", "q": "When was Philippine Advent College founded and who were the founders?", "must_contain": ["1975", "mariano", "margie"]},
    {"cat": "Handbook-Identity", "q": "Who is the school's founder?", "must_contain": ["mariano", "margie"]},
    {"cat": "Handbook-Identity", "q": "Who founded the school?", "must_contain": ["mariano", "margie"]},
    {"cat": "Handbook-Identity", "q": "What does the official PAC logo symbolize?", "must_contain": ["bible", "torch"]},
    {"cat": "Handbook-Identity", "q": "Who composed the music and wrote the lyrics of the PAC Hymn?", "must_contain": ["l.v. pilon", "sl la sage"]},
    {"cat": "Handbook-Identity", "q": "What is the significance of the year 1975 in PAC history?"},
    {"cat": "Handbook-Identity", "q": "What role did the Macias family play in founding PAC?"},

    # ========== 2. Vision / Mission / Values ==========
    {"cat": "Handbook-Vision", "q": "What is the philosophy of education at Philippine Advent College?"},
    {"cat": "Handbook-Vision", "q": "What is PAC’s vision statement?"},
    {
        "cat": "Handbook-Vision",
        "q": "What are the three core values of PAC and what do they mean?",
        "must_contain": ["purpose-driven", "academic excellence", "christ-centered"],
        "must_not_contain": ["they produce goods"],
    },
    {"cat": "Handbook-Vision", "q": "How does PAC describe being Christ-centered?"},
    {"cat": "Handbook-Vision", "q": "Is there a quotation from Ellen G. White’s book Education in the student manual?"},
    {"cat": "Handbook-Conduct", "q": "How many merit points does every student start with each semester?", "must_contain": ["30", "merit"]},
    {"cat": "Handbook-Conduct", "q": "How many demerits is a first-degree offense?", "must_contain": ["5", "demerit"]},
    {"cat": "Handbook-Conduct", "q": "How many demerits is a third-degree offense?", "must_contain": ["30", "demerit"]},
    {"cat": "Handbook-Conduct", "q": "What happens when a student reaches 20 or more demerits?"},
    {"cat": "Handbook-Conduct", "q": "How many days does a student have to appeal a decision of the Student Disciplinary Committee?", "must_contain": ["15", "days"]},
    {"cat": "Handbook-Conduct", "q": "Is jewelry such as rings, bracelets, or necklaces allowed on campus?"},
    {"cat": "Handbook-Conduct", "q": "What is the official dress and appearance policy for students?"},
    {"cat": "Handbook-Conduct", "q": "Are slippers allowed on campus?"},
    {"cat": "Handbook-Conduct", "q": "Is public display of affection considered a violation?", "must_contain_any": ["first-degree", "first degree", "5 demerits"]},
    {"cat": "Handbook-Conduct", "q": "What is the policy on missing dormitory worships?"},
    {
        "cat": "Handbook-Conduct",
        "q": "How many demerits for an unexcused dormitory worship absence?",
        "must_contain": ["worship"],
        "must_contain_any": ["two points", "2 points", "two demerit", "2 demerit"],
    },
    {
        "cat": "Handbook-Conduct",
        "q": "What if a student skips dormitory worship without an excuse?",
        "must_contain": ["worship"],
        "must_contain_any": ["two points", "2 points", "two demerit", "2 demerit"],
    },

    # ========== 4. Legal ==========
    {"cat": "Handbook-Legal", "q": "Does the student manual mention the Anti-Bullying Act or RA 10627?"},
    {
        "cat": "Handbook-Legal",
        "q": "What is the Safe Spaces Act and what is its Republic Act number?",
        "must_contain": ["11313"],
        "must_not_contain": ["11180", "athletic programs report act"],
    },
    {"cat": "Handbook-Legal", "q": "What does RA 11313 cover according to the handbook?"},
    {"cat": "Handbook-Legal", "q": "Is hazing allowed or prohibited in PAC?"},

    # ========== 5. Programs / Fees ==========
    {"cat": "Programs-FAQ", "q": "What degree programs does Philippine Advent College offer?"},
    {"cat": "Programs-FAQ", "q": "What senior high school strands are available at PAC?"},
    {"cat": "Programs-FAQ", "q": "Does PAC offer Bachelor of Science in Nursing and Midwifery?"},
    {"cat": "Programs-FAQ", "q": "What TESDA NC II programs are offered?", "must_contain": ["housekeeping", "food and beverage"]},
    {"cat": "Programs-FAQ", "q": "What teacher education programs are available?"},
    {"cat": "Programs-FAQ", "q": "Is Bachelor of Science in Business Administration offered?"},
    {"cat": "Programs-FAQ", "q": "Is there a family scholarship or discount for siblings?"},
    {"cat": "Programs-FAQ", "q": "What are the current dormitory charges?", "must_contain": ["717.99", "938.91", "331.38", "165.69"]},
    {"cat": "Programs-FAQ", "q": "Is a drug test required for incoming first-year students?"},

    # ========== 6. Schedule + immediate follow-ups ==========
    {"cat": "Schedule", "q": "Do we have classes this week?"},
    {"cat": "Follow-up", "q": "Does that apply to all levels?"},
    {"cat": "Follow-up", "q": "College too?"},
    {"cat": "Follow-up", "q": "Elementary also?"},

    {
        "cat": "Schedule",
        "q": "Is there class tomorrow?",
        "must_contain_any": ["no class", "no classes"],
        "must_not_contain": ["attendance guidelines", "twenty percent", "instructor or professor"],
    },
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

    {
        "cat": "Schedule",
        "q": "Any suspension announced?",
        "must_not_contain": ["unauthorized bulletin boards", "posters must be removed"],
    },
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

FALLBACK_PHRASES = [
    "i don't have a confirmed answer",
    "i've saved it so our staff",
    "i've saved your question so our staff",
    "i couldn't find a solid answer",
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


def is_bad_answer(answer: str, question: str = "",
                  must_contain: list[str] | None = None,
                  must_contain_any: list[str] | None = None,
                  must_not_contain: list[str] | None = None) -> bool:
    if not answer or len(answer.strip()) < 30:
        return True
    lower = answer.lower()
    if any(p in lower for p in FALLBACK_PHRASES):
        return True
    if must_contain and not all(term.lower() in lower for term in must_contain):
        return True
    if must_contain_any and not any(term.lower() in lower for term in must_contain_any):
        return True
    if must_not_contain and any(term.lower() in lower for term in must_not_contain):
        return True
    if any(m in lower for m in WRONG_CHUNK_MARKERS):
        return True

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
    if ("residency" in q or "maximum" in q) and "year" in q:
        if "5 year" not in lower and "five year" not in lower and "maximum of 5" not in lower:
            return True
    return False


def classify_answer_outcome(answer: str, question: str = "",
                           must_contain: list[str] | None = None,
                           must_contain_any: list[str] | None = None,
                           must_not_contain: list[str] | None = None) -> str:
    if (answer or "").startswith("[ERROR]"):
        return "request_error"
    lower = (answer or "").lower()
    if any(phrase in lower for phrase in FALLBACK_PHRASES):
        return "fallback"
    if is_bad_answer(
        answer, question, must_contain, must_contain_any, must_not_contain
    ):
        return "incorrect"
    if must_contain or must_contain_any:
        return "verified"
    return "unverified"


def _extract_answer(data) -> str | None:
    """Pull the bot text out of the many shapes the endpoint can return."""
    if data is None:
        return None
    if isinstance(data, str):
        text = data.strip()
        if text and not text.lower().startswith("<!doctype"):
            return text
        return None
    if not isinstance(data, dict):
        return str(data)

    for key in (
        "reply", "answer", "response", "message", "bot_response",
        "text", "content", "html", "bot_message",
    ):
        val = data.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

    # nested under "data"
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in ("reply", "answer", "message", "text"):
            val = nested.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()

    return None


def ask_chatbot(question: str, call_index: int = 0) -> str:
    """
    Try the payload shapes the frontend and older code actually use.
    Prints diagnostic info for the first few calls so you can see the real
    status / body when something is wrong.
    """
    payloads = [
        {"message": question, "session_id": SESSION_ID},
        {"question": question, "session_id": SESSION_ID},
        {"q": question, "session_id": SESSION_ID},
        {"text": question, "session_id": SESSION_ID},
    ]
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    }

    last_error = None
    for payload in payloads:
        try:
            r = requests.post(
                CHATBOT_URL,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if call_index < DEBUG_FIRST_N:
                print(f"  [debug] status={r.status_code} "
                      f"payload_keys={list(payload.keys())} "
                      f"content-type={r.headers.get('Content-Type', '')}")
                if r.status_code >= 400 or call_index == 0:
                    print(f"  [debug] body[:400]={r.text[:400]!r}")

            if r.status_code >= 400:
                last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                continue

            # Prefer JSON
            try:
                data = r.json()
            except Exception:
                text = (r.text or "").strip()
                if text and len(text) > 20 and not text.lower().startswith("<!doctype"):
                    return text
                last_error = f"Non-JSON body: {text[:200]}"
                continue

            answer = _extract_answer(data)
            if answer:
                return answer

            last_error = (
                f"JSON keys={list(data.keys()) if isinstance(data, dict) else type(data)} "
                f"body={str(data)[:200]}"
            )

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error — is the app running on {CHATBOT_URL}? ({e})"
            break  # no point trying other payloads
        except Exception as e:
            last_error = str(e)

    return f"[ERROR] {last_error}"


def main():
    results: list[dict[str, Any]] = []
    print(f"\nStarting MIXED test — {len(TEST_CASES)} questions")
    print(f"Session: {SESSION_ID}")
    print(f"Endpoint: {CHATBOT_URL}")
    print("=" * 80)

    current_cat = None
    for i, item in enumerate(TEST_CASES, 1):
        cat = item["cat"]
        q = item["q"]
        if cat != current_cat:
            print(f"\n{'─' * 28} {cat} {'─' * 28}")
            current_cat = cat

        print(f"\n[{i:02d}/{len(TEST_CASES)}] {q}")
        answer = ask_chatbot(q, call_index=i - 1)
        outcome = classify_answer_outcome(
            answer, q, item.get("must_contain"), item.get("must_contain_any"),
            item.get("must_not_contain"),
        )
        failed = outcome in {"fallback", "incorrect", "request_error"}
        status = "❌ FAILED" if failed else "✅ PASSED"
        print(f" → {status}")
        print(f" → OUTCOME: {outcome}")
        print(f" → {answer[:240]}{'...' if len(answer) > 240 else ''}")

        results.append({
            "number": i,
            "category": cat,
            "question": q,
            "answer": answer,
            "status": "failed" if failed else "passed",
            "outcome": outcome,
            "verification": (
                "explicit" if item.get("must_contain") or item.get("must_contain_any")
                else "guardrail_only" if item.get("must_not_contain") else "basic_only"
            ),
            "timestamp": datetime.now().isoformat(),
        })
        time.sleep(DELAY_BETWEEN_QUESTIONS)

    outcome_counts: Counter[str] = Counter(r["outcome"] for r in results)
    by_cat: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for r in results:
        category = by_cat[r["category"]]
        category["total"] += 1
        category[r["outcome"]] += 1
        if r["status"] == "passed":
            category["passed"] += 1

    answered = sum(1 for r in results if r["status"] == "passed")
    explicit_cases = sum(1 for r in results if r["verification"] == "explicit")
    explicit_verified = sum(
        1 for r in results
        if r["verification"] == "explicit" and r["outcome"] == "verified"
    )

    with open("test_results_mixed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open("test_results_mixed.md", "w", encoding="utf-8") as f:
        f.write("# Chatbot Mixed Test Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Session:** {SESSION_ID}\n\n")
        f.write(f"**Basic checks passed:** {answered}/{len(results)}\n\n")
        f.write(f"**Explicitly verified:** {explicit_verified}/{explicit_cases}\n\n")
        f.write("## Outcome Breakdown\n\n")
        for outcome in ("verified", "unverified", "fallback", "incorrect", "request_error"):
            f.write(f"- **{outcome.replace('_', ' ').title()}**: {outcome_counts[outcome]}\n")
        f.write("\nAnswers without explicit expectations are unverified, not counted as factual verification.\n\n")
        f.write("## Outcomes by Category\n\n")
        for cat, stats in by_cat.items():
            f.write(
                f"- **{cat}**: {stats['passed']}/{stats['total']} passed; "
                f"verified {stats['verified']}, unverified {stats['unverified']}, "
                f"fallback {stats['fallback']}, incorrect {stats['incorrect']}, "
                f"request errors {stats['request_error']}\n"
            )
        f.write("\n---\n\n")
        for r in results:
            icons = {
                "verified": "✅", "unverified": "⚪", "fallback": "↩️",
                "incorrect": "❌", "request_error": "⚠️",
            }
            icon = icons[r["outcome"]]
            f.write(
                f"### {icon} Q{r['number']} [{r['category']}] "
                f"({r['outcome']}, {r['verification']}): {r['question']}\n\n"
            )
            f.write(f"**Answer:**\n\n{r['answer']}\n\n---\n\n")

    print("\n" + "=" * 80)
    print(f"Finished! Overall: {answered}/{len(results)} passed")
    print("By category:")
    for cat, stats in by_cat.items():
        print(f"  {cat:20} {stats['passed']:2}/{stats['total']}")
    print("Outcome counts:", dict(outcome_counts))
    print(f"Explicitly verified: {explicit_verified}/{explicit_cases}")
    print("\nWrote test_results_mixed.json and test_results_mixed.md")


if __name__ == "__main__":
    main()