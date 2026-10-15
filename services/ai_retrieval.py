"""
services/ai_retrieval.py — AI-first content routing.

THE PROBLEM THIS REPLACES
--------------------------
handbook.py's get_top_handbook_chunks() and services/topic_aliases.py's
regex `match_pattern` / `title_needles` / `content_needles` require a
human to hand-write (and constantly extend) a pattern for every phrasing
of every topic. That's why handbook.py has a long block of one-off
`if re.search(...)` boosts, each one added to fix ONE failing test
question -- and it still misses anything nobody happened to write a
pattern for (see the mixed-test report: jewelry, slippers, the dorm
worship policy, the degree-program list, sibling scholarships, "is there
a suspension announced", etc. all lost to an unrelated chunk that
happened to share more keywords than the right one).

This module replaces "write a pattern" with a single LLM call that reads
the *question* against the *live list of section/FAQ titles* and picks
the one(s) that answer it -- the same way a staff member skimming a
table of contents would. Nothing here is per-topic or per-question: add
a new handbook chapter or FAQ tomorrow and it's covered immediately,
because the router sees the CURRENT title list on every call, not a
hardcoded list someone has to maintain.

This does NOT replace keyword/embedding retrieval or the grounding
check -- it is an additional high-trust ranking signal that sits ON TOP
of whatever candidates the existing pipeline already assembles, and
every answer built from its picks still goes through
services.grounding.answer_is_grounded before being trusted. If the LLM
is unavailable, rate-limited, or genuinely unsure, every function here
returns an empty result ("no opinion") rather than guessing -- callers
must treat that as "fall back to the existing scoring", never as
"nothing matches".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional, Sequence, cast

from ai_providers import ai_cache, circuit_breaker, llm_chat

logger = logging.getLogger("pac_helpdesk.ai_retrieval")

MAX_ITEMS_PER_CALL = 500


def _strip_code_fence(text: str) -> str:
    return re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()


def _corpus_fingerprint(items: Sequence[str]) -> int:
    """Cheap fingerprint so the response cache automatically invalidates
    itself the moment the title list changes (new/edited/removed
    section or FAQ) -- nobody has to remember to bust a cache when they
    add content; the cache key changes on its own."""
    return hash(tuple(items))


def _parse_index_array(raw: str, n: int, max_picks: int) -> List[int]:
    raw = _strip_code_fence(raw)
    try:
        idxs = json.loads(raw)
    except Exception:
        m = re.search(r'\[[^\]]*\]', raw)
        if not m:
            return []
        try:
            idxs = json.loads(m.group(0))
        except Exception:
            return []
    if not isinstance(idxs, list):
        return []
    parsed_idxs: Sequence[object] = cast(Sequence[object], idxs)
    out: List[int] = []
    for item in parsed_idxs:
        if not isinstance(item, (int, float, str)):
            continue
        try:
            i = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in out:
            out.append(i)
        if len(out) >= max_picks:
            break
    return out


def llm_pick_sections(question: str, titles: Sequence[str], max_picks: int = 3,
                       context_hint: Optional[str] = None) -> List[str]:
    """Given the FULL current list of handbook section titles, ask the
    LLM which one(s) actually answer the question. Returns [] if
    unavailable, unsure, or nothing plausibly fits -- treat that as "no
    opinion", not "nothing matches" (the caller's existing keyword/
    embedding scoring still runs regardless).

    This is what makes newly-ingested handbook sections work immediately
    without anyone writing a topic_aliases pattern for them: the titles
    list passed in is whatever's currently in the DB, not a hardcoded
    seed list.

    `context_hint`: optional short text describing what the conversation
    was already about (e.g. the previous turn's question/answer, or the
    section(s) it was answered from) -- lets a short follow-up like "does
    that apply to all levels?" be routed correctly even though it has
    almost no content words of its own to match against. Purely additive:
    omit it and this behaves exactly as a standalone-question router.
    """
    titles = [t for t in dict.fromkeys(t for t in titles if t)]  
    if not question or not titles or circuit_breaker.is_open():
        return []
    titles_used = titles[:MAX_ITEMS_PER_CALL]

    cache_key = (
        f"secpick_{hash(question)}_{_corpus_fingerprint(titles_used)}_"
        f"{max_picks}_{hash(context_hint or '')}"
    )
    cache: Any = ai_cache
    cached = cast(Optional[List[str]], cache.get(cache_key))
    if cached is not None:
        return cached

    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles_used))
    context_block = (
        f"\nContext -- what the conversation was already about (the student's new "
        f"question may be a short follow-up that depends on this):\n{context_hint}\n"
        if context_hint else ""
    )
    prompt = f"""You are matching a student's question to the section(s) of a school
handbook's table of contents most likely to contain the answer.
{context_block}
Student question: "{question}"

Table of contents (index. section title):
{numbered}

Think about what the student is actually asking (using the context above if the
question is a short follow-up that depends on it), then reply with ONLY a JSON
array of the index numbers (0-based) of the section(s) that would answer it,
best match first, at most {max_picks} item(s).
If NONE of the sections plausibly answer this question, reply with an empty
array: []
Reply with the JSON array and nothing else -- no explanation, no markdown.

JSON array:"""

    try:
        resp = llm_chat(prompt, max_tokens=60)
        if not resp:
            return []
        picks_idx = _parse_index_array(resp, len(titles_used), max_picks)
        picks = [titles_used[i] for i in picks_idx]
        cast(Any, ai_cache).set(cache_key, picks)
        return picks
    except Exception as e:
        logger.warning(f"llm_pick_sections error: {e}")
        return []


def llm_pick_faq(question: str, faq_questions: Sequence[str], max_picks: int = 1) -> List[int]:
    """Same idea for FAQ selection. `faq_questions` is normally an
    already-narrowed shortlist (e.g. the output of
    answer_engine.get_top_faq_candidates, ~5 items) rather than the
    entire FAQ table, since answer_engine already does cheap fuzzy/
    embedding narrowing first and this is the tie-breaker on top of
    that -- but it works with a larger list too, up to MAX_ITEMS_PER_CALL.

    Returns the INDEX/indices into `faq_questions` of genuine matches, or
    [] if genuinely unsure -- callers should fall back to their existing
    disambiguation-picker UX when this comes back empty, not treat empty
    as "no FAQ matches at all".
    """
    faq_questions = list(faq_questions)
    if not question or not faq_questions or circuit_breaker.is_open():
        return []
    faq_questions_used = faq_questions[:MAX_ITEMS_PER_CALL]

    cache_key = f"faqpick_{hash(question)}_{_corpus_fingerprint(faq_questions_used)}_{max_picks}"
    cached = cast(Optional[List[int]], cast(Any, ai_cache).get(cache_key))
    if cached is not None:
        return cached

    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(faq_questions_used))
    prompt = f"""A student asked: "{question}"

Here are candidate FAQ questions that might match:
{numbered}

Reply with ONLY a JSON array of the index number(s) (0-based) of the FAQ(s)
that genuinely answer the student's question, best match first, at most
{max_picks} item(s). If you are not confident any of them are a real match,
reply with an empty array: []
Reply with the JSON array and nothing else -- no explanation, no markdown.

JSON array:"""

    try:
        resp = llm_chat(prompt, max_tokens=40)
        if not resp:
            return []
        picks = _parse_index_array(resp, len(faq_questions_used), max_picks)
        cast(Any, ai_cache).set(cache_key, picks)
        return picks
    except Exception as e:
        logger.warning(f"llm_pick_faq error: {e}")
        return []


def llm_self_check(question: str, answer: str, source_excerpt: str) -> bool:
    """Final cheap self-critique pass: "does this answer actually address
    what was asked, using only this excerpt?" -- a second, independent
    opinion that sits ON TOP of services.grounding.answer_is_grounded
    (checks the answer against its source) and
    answer_engine._answer_relevant_to_question /
    handbook._answer_relevant_to_question (checks the answer against the
    question's own words). Those two are cheap word-overlap heuristics;
    this one actually reads the question and the answer and judges fit,
    which catches the cases word-overlap can miss -- e.g. an answer that
    shares plenty of words with the question but actually dodges it, or
    answers a narrower/broader version of what was asked.

    Returns True only when the model explicitly replies YES. An unavailable
    or ambiguous check rejects the generated answer so callers can use their
    deterministic excerpt fallback instead of trusting an unchecked answer.
    """
    if not question or not answer:
        return False
    if circuit_breaker.is_open():
        return False
    try:
        prompt = f"""Student question: "{question}"

Proposed answer: "{answer[:600]}"

Source excerpt the answer must be based on: "{(source_excerpt or '')[:1200]}"

    Does EVERY factual sentence in the proposed answer directly help answer the
    student's question? The excerpt may contain multiple topics: do not approve an
    answer merely because an extra fact appears somewhere in the excerpt. Reply NO
    if the answer includes any unrelated topic, law, number, policy, or side fact,
    even when that fact is accurate and present in the excerpt. Also reply NO if an
    answer claim is not supported by the excerpt. Reply with exactly one word:
    YES or NO."""
        resp = llm_chat(prompt, max_tokens=5)
        if not resp:
            return False
        verdict = resp.strip().upper()
        if verdict != 'YES':
            logger.info(f"llm_self_check: REJECTED answer for q={question[:80]!r}")
            return False
        return True
    except Exception as e:
        logger.warning(f"llm_self_check error: {e}")
        return False