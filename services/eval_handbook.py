#!/usr/bin/env python3
"""
Offline handbook ranking eval against handbook_chunks_pac_2025.json.

Does NOT need a live DB or LLM — checks that forced/alias matching picks
the correct section title for each known test question.

Usage:
  python eval_handbook.py path/to/handbook_chunks_pac_2025.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Prefer the services module if on PYTHONPATH
try:
    from services.topic_aliases import forced_topic_for
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from topic_aliases import forced_topic_for


TESTS = [
    ("What does PAC want to become in the future?", "Vision"),
    ("What are the core values again? The PAC ones.", "Core Values"),
    ("What's the heaviest demerit level?", "Third-Degree"),
    ("How bad is a second-degree offense?", "Second-Degree"),
    ("What's a first degree offense worth in demerits?", "First-Degree"),
    ("What counts as a serious or third-degree offense?", "Third-Degree"),
    ("Is there a maximum fine for violations?", "Disciplinary Measures"),
    ("How long do I have to appeal a disciplinary decision?", "Prescription and Appeal"),
    ("Is it okay to go on hikes without a chaperone?", "First-Degree"),
    ("Are we allowed to visit other dorms freely?", "First-Degree"),
    ("Can we cook inside the dorm?", "Second-Degree"),
    ("What food is not allowed in the cafeteria?", "Second-Degree"),
    ("Is there an anti-bullying law in the handbook?", "GRIEVANCE"),
    ("What's the Safe Spaces Act and its RA number?", "Gender-Based"),
    ("What programs or courses does PAC actually offer?", "Academic Program"),
    ("What senior high strands are available?", "Academic Program"),
    ("What's the main mission of the college?", "Mission"),
    ("What's the school's philosophy about education?", "Philosophy"),
    ("Can we wear jewelry like rings or necklaces?", "First-Degree"),
]


def pick(chunks, question):
    hit = forced_topic_for(question)
    if not hit:
        return []
    titles, needles = hit
    title_hits, content_hits = [], []
    for c in chunks:
        tl = (c.get("section_title") or "").lower()
        blob = f"{tl} {(c.get('content') or '')}".lower()
        if any(n.lower() in tl for n in titles):
            title_hits.append(c["section_title"])
        elif any(n.lower() in blob for n in needles):
            content_hits.append(c["section_title"])
    return title_hits[:2] or content_hits[:2]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "handbook_chunks_pac_2025.json"
    data = json.loads(Path(path).read_text())
    chunks = data["chunks"] if isinstance(data, dict) else data

    ok = 0
    for q, expect in TESTS:
        hits = pick(chunks, q)
        matched = any(expect.lower() in h.lower() for h in hits)
        status = "PASS" if matched else "FAIL"
        if matched:
            ok += 1
        print(f"{status:4}  expect~{expect:28}  got={hits[:1]}")
        if not matched:
            print(f"       Q: {q}")
    print(f"\n{ok}/{len(TESTS)} passed")
    sys.exit(0 if ok == len(TESTS) else 1)


if __name__ == "__main__":
    main()
