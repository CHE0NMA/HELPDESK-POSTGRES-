print("=" * 70)
print("DIAGNOSE 2 — Testing full handbook answer path")
print("=" * 70)

from services import handbook as handbook_service
from services import language as lang_service

questions = [
    "What is the vision of Philippine Advent College?",
    "What is the philosophy of Philippine Advent College?",
    "How many demerits is a First-Degree Offense?",
    "What is the school's tagline or motto?",
]

for q in questions:
    print(f"\n{'='*70}")
    print(f"QUESTION: {q}")
    print("-" * 70)

    # 1. Classification
    classification = lang_service.classify_question_ai(q)
    print(f"1. Classification → {classification}")

    # 2. Is it detected as handbook question?
    is_hb = handbook_service.is_handbook_question(q)
    print(f"2. is_handbook_question → {is_hb}")

    # 3. Retrieval
    chunks = handbook_service.get_top_handbook_chunks(q)
    print(f"3. Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks[:3], 1):
        print(f"   {i}. Score={c['score']:.3f} | {c['section_title']} (p.{c['page_start']})")
        print(f"      Preview: {c['content'][:120]}...")

    # 4. Try to generate answer
    print("4. Trying answer_handbook_question()...")
    try:
        answer = handbook_service.answer_handbook_question(q, detected_lang='english')
        print(f"   Answer → {answer[:300] if answer else None}")
    except Exception as e:
        print(f"   ERROR → {e}")