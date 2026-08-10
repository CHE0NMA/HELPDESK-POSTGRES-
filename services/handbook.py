import logging
import re
from typing import Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz

from ai_providers import ai_cache, circuit_breaker, llm_chat
from database import get_db
from services import nlg
from services.disambiguation import (
    HANDBOOK_POLICY,
    select_contenders,
    should_disambiguate,
)
from services.grounding import answer_is_grounded, combined_source_text
from services.rerank import get_idf_index, rarity_rerank
from utils.text import fuzzy_score, normalize_input

logger = logging.getLogger("handbook_qa")

HANDBOOK_KEYWORDS = [
    'handbook', 'manual', 'rule', 'rules', 'regulation', 'policy', 'policies',
    'code of conduct', 'behavioral code', 'discipline', 'disciplinary',
    'violation', 'penalty', 'sanction', 'offense', 'offence',
    'dress code', 'uniform', 'attire', 'appearance', 'jewelry', 'jewellery',
    'attendance', 'absent', 'absence', 'tardiness', 'tardy',
    'suspension', 'expulsion', 'grounds for', 'prohibited', 'allowed',
    'fraternity', 'sorority', 'hazing', 'bullying',
    'vision', 'mission', 'philosophy', 'core values', 'values',
    'merit', 'demerit', 'points', 'curfew', 'dormitory', 'dorm',
    'motto', 'tagline', 'slogan', 'campus', 'campuses',
    'founded', 'founder', 'history', 'original name',
    'logo', 'emblem', 'symbol', 'hymn', 'alma mater',
    'pwede ba', 'bawal', 'patakaran', 'gawi', 'uniforme', 'damit',
    'absent', 'pagliban', 'disiplina', 'sumbong',
]

_STOP = {
    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'are', 'was', 'were',
    'for', 'on', 'at', 'by', 'with', 'as', 'that', 'this', 'it', 'be', 'what',
    'when', 'where', 'who', 'how', 'does', 'did', 'do', 'its', 'their',
}

_GENERIC = {
    'student', 'students', 'school', 'college', 'handbook', 'manual',
    'section', 'page', 'according', 'philippine', 'advent', 'pac',
}

_QUERY_EXPANSIONS = [
    (r'\b(light|minor)\s+offenses?\b', 'first degree demerit offense'),
    (r'\b(medium|moderate)\s+offenses?\b', 'second degree demerit offense'),
    (r'\b(serious|heavy|heaviest|worst)\s+offenses?\b', 'third degree demerit offense'),
    (r'\b(called before|original name|formerly known|old name)\b', 'originally known founded history hillside view'),
    (r'\b(want to become|aspire to be|future of the (school|college)|what does .+ want)\b', 'vision envisions center of excellence'),
    (r'\b(start with|how many points|points do we start|merit points)\b', 'merit points demerit 30 points'),
    (r'\b(can we wear|allowed to wear|jewelry|necklace|ring)\b', 'dress appearance jewelry wedding ring'),
    (r'\b(hike|outing).{0,20}(chaperone|without)\b', 'unchaperoned hike outing first degree'),
    (r'\b(too many|accumulate|reach|over|more than|\d+\+?)\s*.{0,15}(demerits?)\b', 'probation disciplinary measures 20 demerits'),
    (r'\b(philosophy|school.?s? philosophy|philosophy about)\b', 'philosophy god is the source true education'),
    (r'\b(mission of|main mission|what.?s the mission)\b', 'mission useful human service personal integrity'),
    (r'\b(core values|pac values|the pac ones)\b', 'core values purpose-driven academic excellence christ-centered'),
    (r'\b(logo|emblem|symbol|what does .+ logo)\b', 'logo symbols bible torch 1975'),
    (r'\b(hymn|who wrote|composer|lyrics)\b', 'hymn pilon la sage alma mater'),
    (r'\b(founded|when did .+ start|who founded|founder)\b', 'founded 1975 macias hillside view history'),
    (r'\b(full name|real name|name of (this )?school|campuses)\b', 'philippine advent college inc campuses'),
    (r'\b(anti.?bullying|bullying law|ra.?10627)\b', 'anti-bullying ra 10627'),
    (r'\b(safe spaces|ra.?11313)\b', 'safe spaces act ra 11313'),
    (r'\b(dress code|how (should|do) (i|we) dress)\b', 'dress and appearance uniform attire'),
    (r'\b(cook|cooking).{0,15}(dorm|dormitory)\b', 'dormitory cooking prohibited rice cooker'),
    (r'\b(visit|visiting).{0,15}(other )?dorms?\b', 'dormitory visit other dorms'),
    (r'\b(senior high|strands?|shs)\b', 'senior high stem humss gas abm grades'),
    (r'\b(programs?|courses?).{0,20}(offer|available)\b', 'bachelor programs education business computer'),
    (r'\b(heaviest|worst|most serious).{0,20}(demerit|offense|level)\b', 'third degree offense 30 demerits'),
    (r'\b(maximum fine|max fine|fine for)\b', 'fine 1500 pesos demerit'),
    (r'\b(appeal).{0,30}(disciplinary|decision|decision)\b', 'appeal 15 days prescription'),
]


def _content_words(text: str) -> set:
    words = re.findall(r"[a-z0-9']{3,}", (text or '').lower())
    return {w for w in words if w not in _STOP}


def expand_query_for_retrieval(question: str) -> str:
    q = question
    extra = []
    for pattern, terms in _QUERY_EXPANSIONS:
        if re.search(pattern, q, re.I):
            extra.append(terms)
    if extra:
        return question + ' ' + ' '.join(extra)
    return question


def is_handbook_question(question: str) -> bool:
    q = question.lower()
    if any(kw in q for kw in HANDBOOK_KEYWORDS):
        return True
    if re.search(r'\b(minimum|maximum|how many|how much).{0,50}(absent|absence|tardy|late|demerit|merit|point|day)\b', q):
        return True
    if re.search(r'\b(dress|uniform|attire|clothing|jewelry)\s*(code|rule|policy)?\b', q):
        return True
    if re.search(r'\b(vision|mission|philosophy|motto|tagline|history|founded|logo|hymn)\b', q):
        return True
    if re.search(r'\b(full name|name of the (school|college|university)|campuses|original name)\b', q):
        return True
    if any(re.search(p, q, re.I) for p, _ in _QUERY_EXPANSIONS):
        return True
    return False


def _load_all_chunk_texts() -> List[str]:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT section_title, content
            FROM handbook_chunks
            WHERE is_active = 1
        """)
        return [f"{(r[0] or '')} {(r[1] or '')}" for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"_load_all_chunk_texts error: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def _title_similarity(question: str, title: str) -> float:
    if not title or not question:
        return 0.0
    q = question.lower().strip()
    t = title.lower().strip()
    return max(
        fuzz.token_set_ratio(q, t),
        fuzz.partial_ratio(q, t),
        fuzz.token_sort_ratio(q, t),
        fuzz.ratio(q, t),
    ) / 100.0


def get_top_handbook_chunks(question: str, max_chunks: int = 8) -> List[Dict]:
    conn = get_db()
    cur = conn.cursor()
    try:
        search_q = expand_query_for_retrieval(question)
        q_norm = normalize_input(question)
        q_words = _content_words(search_q)

        rows_by_id: Dict[int, tuple] = {}
        try:
            cur.execute("""
                SELECT id, chapter, section_title, page_start, page_end, content,
                       ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS relevance
                FROM handbook_chunks
                WHERE is_active = 1
                  AND search_vector @@ plainto_tsquery('english', %s)
                ORDER BY relevance DESC
                LIMIT %s
            """, (search_q, search_q, max_chunks * 8))
            for r in cur.fetchall():
                rows_by_id[int(r[0])] = r
        except Exception as e:
            logger.warning(f"handbook fulltext query failed, falling back to fuzzy: {e}")
            conn.rollback()

        try:
            cur.execute("""
                SELECT id, chapter, section_title, page_start, page_end, content, 0.0
                FROM handbook_chunks WHERE is_active = 1
            """)
            all_rows = list(cur.fetchall())
        except Exception as e:
            logger.error(f"handbook title scan failed: {e}")
            all_rows = []

        title_hits = []
        for r in all_rows:
            best_ts = max(
                _title_similarity(question, r[2] or ""),
                _title_similarity(search_q, r[2] or ""),
            )
            content_l = (r[5] or "")[:900].lower()
            for w in list(q_words)[:12]:
                if len(w) >= 4 and w in content_l:
                    best_ts = max(best_ts, 0.38)
                    break
            if best_ts >= 0.40 or (len((r[2] or "").split()) <= 6 and best_ts >= 0.28):
                title_hits.append((best_ts, r))
        title_hits.sort(key=lambda x: x[0], reverse=True)
        for best_ts, r in title_hits[: max_chunks * 4]:
            hid = int(r[0])
            if hid not in rows_by_id:
                rows_by_id[hid] = (r[0], r[1], r[2], r[3], r[4], r[5], best_ts * 0.18)

        if not rows_by_id:
            scored = []
            for r in all_rows:
                hid, chapter, title, ps, pe, content = r[0], r[1], r[2], r[3], r[4], r[5]
                blob = f"{title} {content[:600]}"
                score = float(fuzzy_score(q_norm, normalize_input(blob)))
                scored.append({
                    'id': hid, 'chapter': chapter, 'section_title': title,
                    'page_start': ps, 'page_end': pe, 'content': content,
                    'score': score,
                })
            scored.sort(key=lambda x: x['score'], reverse=True)
            return [s for s in scored[:max_chunks] if s['score'] >= 0.28]

        scored: List[Dict] = []
        for r in rows_by_id.values():
            fts = float(r[6]) if r[6] else 0.05
            title = (r[2] or "")
            title_l = title.lower()
            content = r[5] or ""
            content_preview = content[:400].lower()
            blob = f"{title_l} {content[:500]}".lower()

            title_sim = max(
                _title_similarity(question, title),
                _title_similarity(search_q, title),
            )
            kw_hits = sum(1 for w in q_words if len(w) >= 4 and w in blob)
            if kw_hits >= 2:
                title_sim = max(title_sim, 0.42)

            n_chars = len(content)
            if title_sim >= 0.50 and n_chars < 900:
                length_factor = 2.0
            elif title_sim >= 0.35 and n_chars < 1800:
                length_factor = 1.4
            elif n_chars > 4000:
                length_factor = 0.65
            else:
                length_factor = 1.0

            overlap = sum(1 for w in q_words if w in blob)
            overlap_factor = 1.0 + (0.12 * min(overlap, 6)) if overlap else 1.0

            noise = 1.0
            if "table of contents" in title_l or title_l.strip() in ("introduction", "general"):
                noise = 0.12
            elif "definition of terms" in title_l:
                noise = 0.35
            if title_l.strip() == "mission" and "step 5" in content_preview:
                noise = 0.08

            score = (
                (fts + 0.05)
                * (0.35 + 1.8 * title_sim)
                * length_factor
                * overlap_factor
                * noise
            )
            scored.append({
                'id': r[0], 'chapter': r[1], 'section_title': r[2],
                'page_start': r[3], 'page_end': r[4], 'content': content,
                'score': score,
            })

        try:
            idf = get_idf_index(_load_all_chunk_texts)
            scored = rarity_rerank(
                question, scored, idf,
                text_fn=lambda c: f"{c.get('section_title', '')} {c.get('content', '')}",
                score_key='score', blend=0.12,
            )
        except Exception as e:
            logger.warning(f"rarity_rerank skipped: {e}")

        scored.sort(key=lambda x: x['score'], reverse=True)
        top = float(scored[0]['score'] or 1.0)
        for s in scored:
            s['score'] = float(s['score']) / top if top else 0.0
        return scored[:max_chunks]

    except Exception as e:
        logger.error(f"get_top_handbook_chunks error: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def _page_ref(chunk: Dict) -> str:
    ps, pe = chunk.get('page_start'), chunk.get('page_end')
    return f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"


def build_handbook_disambiguation(chunks: List[Dict], detected_lang: str = 'english') -> Optional[str]:
    if not chunks:
        return "I'm not sure what you're asking about in the handbook. Could you rephrase?"
    headers = {
        'english': "I found a few handbook sections that might be relevant. Which one are you asking about?",
        'tagalog': "May ilang seksyon ako sa handbook na maaaring may kinalaman. Alin dito ang tinutukoy mo?",
        'bisaya': "Naa koy pipila ka seksyon sa handbook nga posibleng may kalabutan. Asa niini ang imong gipangita?",
    }
    header = headers.get(detected_lang, headers['english'])
    scores = [c.get('score', 0.0) for c in chunks]
    contenders = select_contenders(chunks, scores, HANDBOOK_POLICY)
    if len(contenders) <= 1:
        return None
    options = []
    for i, c in enumerate(contenders, 1):
        title = re.sub(r'\s*\(part\s*\d+\)\s*$', '', c.get('section_title') or '', flags=re.I).strip()
        options.append(f"{i}. {title} ({_page_ref(c)})")
    return f"{header}\n\n" + "\n".join(options)


def build_handbook_disambiguation_options(chunks: List[Dict]) -> List[Dict]:
    scores = [c.get('score', 0.0) for c in chunks]
    contenders = select_contenders(chunks, scores, HANDBOOK_POLICY)
    return [
        {
            'label': f"{re.sub(r'\\s*\\(part\\s*\\d+\\)\\s*$', '', c.get('section_title') or '', flags=re.I).strip()} ({_page_ref(c)})",
            'chunk': c,
        }
        for c in contenders
    ]


def _lead_text(chunk: Dict, max_chars: int = 400) -> str:
    title = (chunk.get('section_title') or '').strip()
    content = (chunk.get('content') or '').strip()
    parts = re.split(r'(?<=[.!?])\s+', content)
    lead = ' '.join(p.strip() for p in parts[:2] if p.strip())
    if not lead:
        lead = content[:max_chars]
    return f"{title}. {lead}"[:max_chars]


def _content_word_overlap_score(question: str, chunk: Dict) -> Tuple[int, int]:
    q_words = _content_words(expand_query_for_retrieval(question)) - _GENERIC
    if not q_words:
        return 0, 1
    blob = f"{chunk.get('section_title', '')} {chunk.get('content', '')}".lower()
    hits = sum(1 for w in q_words if w in blob)
    required = max(1, min(3, len(q_words) // 3))
    return hits, required


def _title_match_score(question: str, chunk: Dict) -> float:
    title = chunk.get('section_title') or ''
    return max(
        _title_similarity(question, title),
        _title_similarity(expand_query_for_retrieval(question), title),
    )


def _lead_fuzzy_score(question: str, chunk: Dict) -> float:
    lead = _lead_text(chunk)
    return max(
        fuzz.token_set_ratio(question.lower(), lead.lower()),
        fuzz.partial_ratio(question.lower(), lead.lower()),
    ) / 100.0


def llm_hard_relevance(question: str, chunk: Dict) -> bool:
    if circuit_breaker.is_open():
        return False
    title = chunk.get('section_title') or ''
    content = (chunk.get('content') or '')[:1200]
    prompt = (
        f"Does this handbook section answer the student question?\n"
        f"Question: {question}\n"
        f"Section title: {title}\n"
        f"Content: {content}\n"
        f"Reply YES or NO only."
    )
    try:
        resp = (llm_chat(prompt, max_tokens=8) or '').strip().upper()
        return resp.startswith('YES')
    except Exception as e:
        logger.warning(f"llm_hard_relevance error: {e}")
        return False


def chunk_is_relevant_hard(question: str, chunk: Dict) -> bool:
    hits, required = _content_word_overlap_score(question, chunk)
    title_score = _title_match_score(question, chunk)
    expanded = expand_query_for_retrieval(question)
    title_score = max(
        title_score,
        _title_similarity(expanded, chunk.get('section_title') or ''),
    )
    lead_score = _lead_fuzzy_score(question, chunk)

    exp_words = _content_words(expanded) - _GENERIC
    blob = f"{chunk.get('section_title', '')} {chunk.get('content', '')[:900]}".lower()
    exp_hits = sum(1 for w in exp_words if w in blob)

    if title_score >= 0.45:
        return True
    if title_score >= 0.32 and (hits >= 1 or exp_hits >= 2):
        return True
    if exp_hits >= 3:
        return True
    if hits >= required and (title_score >= 0.30 or lead_score >= 0.35):
        return True
    if hits >= required + 1 or exp_hits >= required + 1:
        return True
    return False


def get_relevant_chunks(question: str, candidates: List[Dict]) -> List[Dict]:
    if not candidates:
        return []
    hard = [c for c in candidates if chunk_is_relevant_hard(question, c)]
    if hard:
        return hard[:4]
    llm_kept = []
    for c in candidates[:4]:
        if llm_hard_relevance(question, c):
            llm_kept.append(c)
    if llm_kept:
        return llm_kept[:4]
    if candidates and candidates[0].get('score', 0) >= 0.55:
        return candidates[:1]
    return []


def _extract_relevant_sentences(content: str, question: str, max_sentences: int = 5) -> str:
    q_words = _content_words(expand_query_for_retrieval(question))
    sentences = re.split(r'(?<=[.!?])\s+', content or '')
    ranked = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        hits = sum(1 for w in q_words if w in s.lower())
        ranked.append((hits, s))
    ranked.sort(key=lambda x: x[0], reverse=True)
    chosen = [s for h, s in ranked[:max_sentences] if h > 0]
    if not chosen:
        chosen = [s for h, s in ranked[:2]]
    return ' '.join(chosen)


def _answer_relevant_to_question(question: str, answer: str, min_overlap: int = 1) -> bool:
    q_words = _content_words(question) - _GENERIC
    a_words = _content_words(answer)
    if not q_words or not a_words:
        return True
    q_specific = q_words - _GENERIC
    if not q_specific:
        return True
    return len(q_specific & a_words) >= min_overlap


def _build_handbook_answer_from_chunks(question: str, chunks: List[Dict],
                                       detected_lang: str = 'english') -> Optional[str]:
    if not chunks:
        return None

    relevant = get_relevant_chunks(question, chunks)
    if not relevant:
        logger.info(
            "handbook HARD relevance gate rejected all candidates "
            f"for q={question[:80]!r}"
        )
        return None
    chunks = relevant
    source_blob = combined_source_text(chunks)

    if not circuit_breaker.is_open():
        try:
            context_parts = []
            for i, c in enumerate(chunks[:4], 1):
                context_parts.append(
                    f"[Excerpt {i} – {c.get('section_title')} (p.{c.get('page_start')})]\n"
                    f"{c.get('content', '')[:2200]}"
                )
            context = "\n\n".join(context_parts)

            prompt = f"""{nlg.now_context()}

You are a helpful student handbook assistant. Answer the student's question using ONLY the handbook excerpts below.
Do not invent any facts, names, numbers, or rules that are not present in the excerpts.
If the excerpts do not contain enough information to answer the question, reply with exactly: CANNOT_ANSWER

Student question: {question}

Handbook excerpts:
{context}

Instructions:
- Be precise and concise.
- Quote or closely paraphrase the relevant rule.
- If the answer involves numbers (demerits, points, days, fines), state them clearly.
- If it is a yes/no rule, state the rule clearly.
- Cite the section title and page when helpful.
- {nlg.STYLE_GUIDE}

Answer:"""

            response = llm_chat(prompt, max_tokens=450)
            if response and "CANNOT_ANSWER" not in response.upper():
                response = response.strip()
                if not answer_is_grounded(response, source_blob, min_overlap=0.35):
                    logger.info(f"handbook grounding rejected for q={question[:80]!r}")
                elif not _answer_relevant_to_question(question, response, min_overlap=1):
                    logger.info(f"handbook answer-relevance REJECTED for q={question[:80]!r}")
                else:
                    return response
        except Exception as e:
            logger.error(f"AI handbook answer failed: {e}")

    top = chunks[0]
    title = top.get('section_title', 'the Student Handbook')
    page = top.get('page_start')
    content = (top.get('content') or '').strip()
    clean = _extract_relevant_sentences(content, question, max_sentences=4)
    if not clean:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        clean = ' '.join(s.strip() for s in sentences[:3] if len(s.strip()) > 25)
    if not clean or not _answer_relevant_to_question(question, clean):
        return None
    page_note = f" (see page {page})" if page else ""
    return f"According to the section **{title}**{page_note}:\n\n{clean}"


def answer_handbook_question(question: str, detected_lang: str = 'english',
                             min_score_to_answer: Optional[float] = None,
                             ctx=None) -> Optional[str]:
    if circuit_breaker.is_open():
        return None

    cache_key = f"handbook_{hash(question.lower().strip())}"
    cached = ai_cache.get(cache_key)
    if cached:
        return cached

    chunks = get_top_handbook_chunks(question)
    if not chunks:
        return None

    if min_score_to_answer is not None:
        top_score = chunks[0].get('score', 0)
        if 0 <= top_score <= 1 and top_score < min_score_to_answer:
            return None

    if len(chunks) > 1:
        scores = [c.get('score', 0.0) for c in chunks]
        if should_disambiguate(scores, HANDBOOK_POLICY):
            dis = build_handbook_disambiguation(chunks, detected_lang)
            if dis:
                if ctx is not None:
                    ctx.set_disambiguation(
                        'handbook',
                        build_handbook_disambiguation_options(chunks),
                        original_question=question,
                        detected_lang=detected_lang,
                    )
                return dis

    response = _build_handbook_answer_from_chunks(question, chunks[:5], detected_lang)
    if response:
        ai_cache.set(cache_key, response)
        if ctx is not None:
            ctx.clear_disambiguation()
    return response


def answer_from_selected_chunk(original_question: str, chunk: Dict,
                               detected_lang: str = 'english') -> Optional[str]:
    return _build_handbook_answer_from_chunks(original_question, [chunk], detected_lang)
