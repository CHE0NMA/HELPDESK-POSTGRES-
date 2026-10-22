import logging
import re
from typing import Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz

from ai_providers import ai_cache, circuit_breaker, llm_chat
from database import get_db
from services import nlg
from services.ai_retrieval import (
    llm_expand_school_query,
    llm_pick_sections,
    llm_self_check,
)
from services.disambiguation import (
    HANDBOOK_POLICY,
    select_contenders,
    should_disambiguate,
)
from services.embeddings import cosine_similarity
from services.grounding import answer_is_grounded, combined_source_text, grounding_ratio
from services.language import expand_query, normalize_vernacular_spelling, verify_output_language
from services.rerank import get_idf_index, rarity_rerank
from utils.text import fuzzy_score, normalize_input

logger = logging.getLogger("handbook_qa")

HANDBOOK_KEYWORDS = [
    'handbook', 'manual', 'rule', 'rules', 'regulation', 'policy', 'policies',
    'code of conduct', 'behavioral code', 'discipline', 'disciplinary',
    'violation', 'penalty', 'sanction', 'offense', 'offence',
    'dress code', 'uniform', 'attire', 'appearance', 'jewelry', 'jewellery',
    'earring', 'earrings',
    'attendance', 'absent', 'absence', 'tardiness', 'tardy',
    'suspension', 'expulsion', 'grounds for', 'prohibited', 'allowed',
    'fraternity', 'sorority', 'hazing', 'bullying', 'bully', 'bullies', 'bullied',
    'vision', 'mission', 'philosophy', 'core values', 'values',
    'merit', 'demerit', 'points', 'curfew', 'dormitory', 'dorm',
    'motto', 'tagline', 'slogan', 'campus', 'campuses',
    'founded', 'founder', 'history', 'original name',
    'logo', 'emblem', 'symbol', 'hymn', 'alma mater',
    'pwede ba', 'bawal', 'patakaran', 'gawi', 'uniforme', 'damit',
    'absent', 'pagliban', 'disiplina', 'sumbong',
    'suoton', 'isul-ob', 'pananamit', 'damit', 'bisti', 'tsinelas',
    'patakaran', 'bawal', 'pwede ba', 'unsaon', 'unsay dapat',
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

    (r'\b(unsay (dapat )?(suoton|isuot uniform|suot uniform)|'
     r'unsa (ang )?(dapat )?(suoton|isuot uniform|suot uniform|suot)|'
     r'isuot uniform|suot uniform|suoton|sout)\b',
     'dress attire clothing appearance'),
    (r'\b(light|minor)\s+offenses?\b', 'first degree offense demerit'),
    (r'\b(medium|moderate)\s+offenses?\b', 'second degree offense demerit'),
    (r'\b(how bad|how serious|what.?s a|what is a|tell me about).{0,25}second[ -]?degree\b',
     'second degree offense 15 demerits'),
    (r'\bsecond[ -]?degree\s+(offense|offence|demerit)\b',
     'second degree offense 15 demerits'),
    (r'\b(serious|heavy|heaviest|worst)\s+offenses?\b', 'third degree offense demerit'),
    (r'\b(how bad|how serious|what.?s a|what is a|tell me about).{0,25}third[ -]?degree\b',
     'third degree offense 30 demerits'),
    (r'\bthird[ -]?degree\s+(offense|offence|demerit)\b',
     'third degree offense 30 demerits'),
    (r'\b(called before|original name|formerly known|old name)\b', 'originally known history founded'),
    # Vision — cover "want to become", "future", "aspire"
    (r'\b(want to become|aspire to be|future of the (school|college)|what does .+ want|'
     r'envision|what does pac want)\b',
     'vision envisions center of excellence'),
    (r'\b(start with|how many points|points do we start|merit points|how many merits)\b',
     'merit points demerit system'),
    (r'\b(can we wear|allowed to wear|jewelry|necklace|ring)\b', 'dress appearance jewelry'),
    (r'\b(hike|outing).{0,25}(chaperone|without)\b',
     'unchaperoned hike outing first degree offense'),
    (r'\b(too many|accumulate|reach|over|more than|\d+\+?)\s*.{0,15}(demerits?)\b',
     'probation disciplinary demerits'),
    (r'\b(philosophy|school.?s? philosophy|philosophy about)\b', 'philosophy true education'),
    (r'\b(mission of|main mission|what.?s the mission)\b', 'mission human service integrity'),
    (r'\b(core values|pac values|the pac ones|values of (the )?(school|college|pac))\b',
     'core values purpose-driven academic excellence christ-centered'),
    (r'\b(logo|emblem|symbol|what does .+ logo)\b', 'logo symbols bible torch'),
    (r'\b(hymn|who wrote|composer|lyrics)\b', 'hymn alma mater'),
    (r'\b(founded|when did .+ start|who founded|founder|when .{0,12} (built|established))\b',
     'founded history established'),
    (r'\b(full name|real name|name of (this )?school|campuses|other campuses|how many campuses)\b',
     'college campuses campus locations'),
    (r'\b(anti.?bullying|bullying law|ra.?10627)\b', 'anti-bullying bullying ra 10627'),
    (r'\b(safe spaces|ra.?11313|safe spaces act)\b', 'safe spaces act ra 11313 gender harassment'),
    (r'\b(dress code|how (should|do) (i|we) dress)\b', 'dress and appearance uniform attire'),
    (r'\b(cook|cooking).{0,20}(dorm|dormitory|inside)\b',
     'cooking within the dormitory second degree offense demerit'),
    (r'\b(visit|visiting).{0,20}(other )?dorms?\b',
     'visiting in dormitories first degree offense permission dormitory dean'),
    (r'\b(food|cafeteria).{0,30}(not allowed|prohibited|bawal|banned|restricted|allowed)\b',
     'food eating bringing cafeteria not sanctioned sda church second degree'),
    (r'\b(max(imum)?\s*fine|fine for|biggest fine|money.{0,15}fine|is there a maximum fine)\b',
     'disciplinary measures community service fines up to 1500 pesos'),
    (r'\b(senior high|strands?|shs|what strands)\b',
     'senior high stem humss gas abm grades 11 12 academic program offerings'),
    (r'\b(programs?|courses?).{0,20}(offer|available)\b',
     'bachelor programs education business computer academic program offerings'),
    (r'\b(scholarship|scholarships|financial aid|financial assistance)\b',
     'scholarship financial assistance grant stipend'),
    (r'\b(appeal).{0,40}(decision|disciplinary|sanction|how long)\b',
     'prescription and appeal disciplinary days'),
    (r'\b(anti.?bullying|bullying)\b', 'anti-bullying bullying policy third degree'),
    (r'\b(heaviest|worst|most serious).{0,25}(demerit|offense|level)\b',
     'third degree offense 30 demerits'),
    (r'\b(maximum fine|max fine|fine for|biggest (fine|amount)|money for the fine)\b',
     'disciplinary measures fine 1500 pesos'),
    (r'\b(grounds? for|reasons? for).{0,25}(suspension|expulsion|dismissal)\b',
     'suspension expulsion dismissal offense demerit disciplinary behavioral'),
    (r'\b(fraternity|sorority)\b', 'fraternity sorority offense disciplinary'),
    (r'\b(pregnant|pregnancy)\b', 'pregnancy pregnant admission student'),
    (r'\b(how many (degrees?|levels?) (of |are there )?(offense|offence)|'
     r'(offense|offence) (degrees?|levels?)|up to (what|how many) degree)\b',
     'first degree offense second degree offense third degree offense demerits'),
    (r'\b(appeal).{0,30}(disciplinary|decision)\b', 'prescription and appeal disciplinary'),
    (r'\b(facts?|trivia|tell me (about|something about)|know about|'
     r'info(rmation)? about|general info|anything about|what.?s (up )?with)\b'
     r'.{0,25}\b(school|college|pac|institution|campus)\b',
     'vision mission history founded core values campuses philosophy logo hymn'),
    (r'\b(macias|founders?|tell me more about the founders)\b',
     'brief history mariano margie macias founded 1975'),
    (r'\b(something about 1975|year 1975)\b',
     'founded 1975 hillside view logo established'),
    (r'\b(curfew|9:30|curfew hours)\b',
     'curfew 9:30 first degree dormitory residents'),
    (r'\b(pda|public display of affection|going together of sweethearts)\b',
     'public displays of affection first degree offense'),
    (r'\b(miss worship|worship absences|too many times)\b',
     'worship absences dormitory pledge 20 30'),
    (r'\b(community service instead of fine|community service or fine)\b',
     'disciplinary measures community service fines up to 1500'),
    (r'\b(slippers|shorts|short pants|crop tops)\b|'
     r'\b(?:wear|wearing|mag|magsuot|mag\s+suot)\s+short\b',
     'dress attire clothing appearance slippers short pants crop tops '
     'first degree inappropriate attire'),
    (r'\b(questionable music|indecent songs|rules about music)\b',
     'questionable music indecent songs first degree'),
    (r'\b(incomplete grades?|complete an incomplete|deadline for incomplete)\b',
     'incomplete grade must be completed within one year'),
    (r'\b(residency limit|maximum of 5 years|finish a degree)\b',
     'maximum of 5 years to complete the program residency'),
    (r'\b(drug test for first|first.?year drug)\b',
     'first-year students drug test admission'),
    (r'\b(sibling|family scholar|scholarships for siblings)\b',
     'family scholar program four siblings three siblings discount'),
    (r'\b(dormitory fees?|dorm fees?|bed space)\b',
     'dormitory charges old room new room bed space electricity'),
    (r'\b(tesda|housekeeping|caregiving|food and beverage)\b',
     'tesda housekeeping caregiving food and beverage nc ii'),
    (r'\b(it or computer science|computer science|information technology campuses)\b',
     'bachelor of science in information technology computer science main salug'),
    (r'\b(business administration|bsba)\b',
     'bachelor of science in business administration human resource'),
    (r'\b(2025 edition|2025-2026|school year 2025)\b',
     '2025 edition gender-based harassment committee sy 2025-2026'),
    (r'\b(ellen white|education book)\b',
     'ellen g white education greatest want of the world'),
    (r'\b(social media use|rules on social media)\b',
     'responsible use of social media third degree offense'),
    (r'\b(hazing|fraternity|sorority)\b',
     'fraternity sorority hazing never be allowed third degree'),
]


def _content_words(text: str) -> set:
    words = re.findall(r"[a-z0-9']{3,}", (text or '').lower())
    return {w for w in words if w not in _STOP}


def expand_query_for_retrieval(question: str) -> str:

    q = expand_query(question)
    extra = []
    for pattern, terms in _QUERY_EXPANSIONS:
        if re.search(pattern, q, re.I):
            extra.append(terms)
    if extra:
        return q + ' ' + ' '.join(extra)
    return q

_FORCED_TOPIC_MATCHES = [
    (r'\b(want to become|aspire|future of|what does .{0,20} want|envision|envisions)\b',
     ['vision'], ['envisions', 'center of excellence']),
    (r'\b(core values|pac values|the pac ones|values of (the )?(school|college|pac))\b',
     ['core values', 'core value'], ['purpose-driven', 'christ-centered']),
    (r'\b(heaviest|worst|most serious).{0,25}(demerit|offense|level|degree)\b',
     ['third-degree', 'third degree'], ['30 demerit']),
    (r'\bthird[ -]?degree\b',
     ['third-degree', 'third degree'], ['30 demerit']),
    (r'\b(how bad|how serious|what.?s a|what is a).{0,20}second[ -]?degree\b|\bsecond[ -]?degree\b',
     ['second-degree', 'second degree'], ['15 demerit']),
    (r'\b(first[ -]?degree|light\s+offense)\b',
     ['first-degree', 'first degree'], ['5 demerit']),
    (r'\b(max(imum)?\s*fine|is there a maximum fine|biggest fine|fine for)\b',
     ['disciplinary measures'], ['1,500', '1500', 'fines']),
    (r'\b(appeal).{0,50}(decision|disciplinary|sanction|how long|window)|how long.{0,30}appeal|appeal.{0,20}(decision|window)\b',
     ['prescription and appeal'], ['15 days', 'appealed within']),
    (r'\b(hike|outing).{0,30}(chaperone|without|unchaperoned)|unchaperoned|(chaperone|without a chaperone)\b',
     ['first-degree', 'first degree'], ['unchaperoned hike', 'unchaperoned']),
    (r'\b(visit|visiting).{0,25}(other )?dorms?|other dorms?\b',
     ['first-degree', 'first degree'], ['visiting in dormitor', 'visiting in dormitories']),
    (r'\b(cook|cooking).{0,30}(dorm|dormitory|inside)|cook inside\b',
     ['second-degree', 'second degree'], ['cooking within the dormitory', 'cooking within']),
    (r'\b(food|cafeteria).{0,30}(not allowed|prohibited|bawal|banned|restricted)\b',
     ['second-degree', 'second degree'], ['not sanctioned']),
    (r'\b(safe spaces|ra.?11313|safe spaces act|11313)\b',
     ['gender-based sexual harassment', 'gender-based', 'safe spaces'],
     ['11313', 'safe spaces act']),
    (r'\b(anti.?bullying|ra.?10627|bullying law|10627|'
     r'bully|bullies|bullied|bullying|'
     r'nambu-?bully|nanbu-?bully|pambu-?bully|bina-?bully|'
     r'nanintimidar|pagdaugdaug)\b',
     ['grievance', 'anti-bullying', 'bullying', 'third-degree'],
     ['10627', 'anti-bullying', 'bullying']),
    (r'\b(senior high|strands?|shs|what strands)\b',
     ['academic program offerings', 'academic program', 'program offerings'],
     ['stem', 'humss', 'abm', 'grades 11']),
    (r'\b(programs?|courses?).{0,25}(offer|available|does pac)\b',
     ['academic program offerings', 'academic program', 'program offerings'],
     ['bachelor of science', 'bachelor of']),
    (r'\b(main mission|what.?s the mission|mission of)\b',
     ['mission'], ['human service', 'personal integrity']),
    (r'\b(philosophy|school.?s? philosophy)\b',
     ['philosophy'], ['source of true education']),
    (r'\b(jewelry|jewellery|necklace|rings?|earrings?|ear\s*rings?|piercing|'
     r'nose\s*rings?|bracelets?|bangles?|'
     r'(is|are|can|may|okay|ok|allowed).{0,25}(wear|wearing).{0,20}'
     r'(earring|jewelry|jewellery|necklace|ring|bracelet))\b',
     ['dress and appearance', 'first-degree', 'first degree'],
     ['necklace', 'earrings', 'jewelry', 'jewellery', 'wedding ring', 'bracelet']),
    (r'\b(campuses|other campus(es)?|how many campuses|campus locations?|'
     r'where (is|are) the (main )?campus(es)?|which campus(es)?|'
     r'which locations?|names? of the (other )?campuses?|main campus)\b',
     ['meetings'],
     ['held within the campus', 'regular meetings of the members']),
    (r'\b(dress code|how .{0,15}dress|proper attire|uniform|shorts|short pants)\b|'
     r'\b(unsa|unsay).{0,35}\b(isuot|suot|suoton|souton)\b|'
     r'\b(pwede|puwede).{0,30}\b(mag\s+short|magsuot|mag\s+suot)\b|'
     r'\b(?:wear|wearing)\s+short\b',
     ['dress and appearance', 'school uniform', 'dress'],
     ['uniform', 'attire', 'appearance']),
]


def _forced_topic_for(question: str):
    """Return (title_needles, content_needles) if the question is a
    high-precision known topic, else None.

    Prefers data-driven aliases from handbook_topic_aliases (admin-editable).
    Falls back to the in-module _FORCED_TOPIC_MATCHES seed so ranking still
    works before the migration has run.

    `question` is run through vernacular-spelling normalization + hypothetical
    ("what would happen if I ___") unwrapping before any pattern is tried, so
    both the DB-driven aliases and the in-module fallback benefit without
    each pattern needing its own spelling/framing variants hand-written in.
    """
    normalized_q = expand_query(question or '')
    try:
        from services.topic_aliases import forced_topic_for as _db_forced
        hit = _db_forced(normalized_q)
        if hit:
            return hit
    except Exception:
        try:
            from topic_aliases import forced_topic_for as _db_forced
            hit = _db_forced(normalized_q)
            if hit:
                return hit
        except Exception:
            pass
    q = normalized_q.lower()
    for pattern, titles, content_kws in _FORCED_TOPIC_MATCHES:
        if re.search(pattern, q, re.I):
            return titles, content_kws
    return None

_COMPOUND_SPLIT_RE = re.compile(
    r'\s*\b(?:and also|as well as|and)\b\s*'
    r'(?=(?:where|what|when|why|who|whom|which|how|is|are|was|were|do|does|'
    r'did|can|could|should|will|would|may|might)\b)',
    re.I,
)


def split_compound_question(question: str) -> List[str]:
    """Best-effort split of a question that actually bundles two (or more)
    distinct asks into one message, e.g. "What's the real full name of
    this school and where are the campuses?" -> two separate questions,
    each run through the normal single-topic retrieval pipeline instead
    of forcing one query to cover two topics at once.

    Deliberately conservative in both directions it can trigger:
      - Multiple '?' marks -> split on them (rare but unambiguous).
      - A conjunction ("and", "as well as") is only treated as a joint
        ONLY when what follows immediately starts with its own question
        word or auxiliary verb — this avoids shredding ordinary noun-
        phrase "and"s ("rings or necklaces", "food and drinks") into
        nonsense fragments.
    Returns [question] unchanged (a list of one) when nothing that looks
    like a genuine second question is found.
    """
    if not question:
        return [question]
    q = question.strip()

    if q.count('?') >= 2:
        parts = [p.strip() for p in re.split(r'(?<=\?)\s*', q) if p.strip()]
        if len(parts) >= 2:
            return parts

    parts = _COMPOUND_SPLIT_RE.split(q)
    parts = [p.strip(' ?.,') for p in parts if p and p.strip(' ?.,')]
    if len(parts) >= 2:
        return [p if p.endswith('?') else f"{p}?" for p in parts]
    return [q]


def is_handbook_question(question: str) -> bool:
    """Topic-shaped detector only — no edition-specific facts.

    Safe across handbook re-ingests: matches *kinds* of questions the
    student manual covers, not particular years/names/numbers.
    """
    q = normalize_vernacular_spelling(question.lower())
    if any(kw in q for kw in HANDBOOK_KEYWORDS):
        return True
    if re.search(
        r'\b(unsa|unsay|unsaon).{0,35}\b(suoton|isuot|suot|shorts?|damit|'
        r'sinina|bisti|panapton)\b|'
        r'\b(pwede|puwede).{0,30}\b(magsuot|mag\s+suot|mag\s+shorts?|'
        r'short|shorts|wear)\b.{0,35}\b(school|campus|diri|dinhi|diha)?\b|'
        r'\b(shorts?|short pants)\b.{0,30}\b(school|campus|allowed|okay|'
        r'policy|rule|pwede|bawal)\b',
        q, re.I,
    ):
        return True
    if re.search(
        r'\b(what (?:would|will) happen(?:s)? if i|what happens if i|what if i|'
        r'ano (?:ang |po ang )?mangyayari (?:kung|kapag) ako|paano kung ako|'
        r'kung ako(?:y|\s+ay)?|unsa(?:y)? (?:ang )?mahitabo kung ako|'
        r'unsaon (?:kung|kon) ako|kon ako)\b',
        q, re.I,
    ):
        return True
    if re.search(
        r'\b(minimum|maximum|how many|how much).{0,50}'
        r'(absent|absence|tardy|late|demerit|merit|point|day|fine)\b',
        q,
    ):
        return True
    if re.search(r'\b(dress|uniform|attire|clothing|jewelry)\s*(code|rule|policy)?\b', q):
        return True
    if re.search(r'\b(vision|mission|philosophy|motto|tagline|history|founded|logo|hymn)\b', q):
        return True
    if re.search(
        r'\b(full name|name of the (school|college|university)|campuses|'
        r'original name|other campuses|how many campuses)\b',
        q,
    ):
        return True

    if re.search(
        r'\b(grounds? for (suspension|expulsion|dismissal)|'
        r'(suspension|expulsion|dismissal).{0,25}(grounds?|offense|demerit|reason)|'
        r'(first|second|third)[ -]?degree|'
        r'demerit|merit points?|fraternity|sorority|'
        r'pregnant|pregnancy|'
        r'disciplinary (action|measure|committee)|'
        r'code of conduct|student discipline|behavioral code)\b',
        q, re.I,
    ):
        return True
    if re.search(
        r'\b(when (was|is) (pac|the school|the college) '
        r'(built|founded|established)|who founded)\b',
        q, re.I,
    ):
        return True
    if re.search(
        r'\b(dress\s*code|uniform|jewelry|senior high|strands?|'
        r'scholarship|safe spaces|anti.?bullying|appeal|'
        r'full name|real name|programs? (offered|available)|'
        r'chaperone|unchaperoned|want to become|core values|'
        r'maximum fine|max fine|heaviest demerit|cook(ing)? (in|inside)|'
        r'visit other dorms|food .{0,20}(cafeteria|not allowed)|'
        r'how bad is a (second|third)|what.?s a (second|third)[ -]?degree|'
        r'ra.?11313|ra.?10627|safe spaces act)\b',
        q, re.I,
    ):
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


def _get_top_handbook_chunks(
    question: str,
    max_chunks: int = 8,
    query_embedding: Optional[List[float]] = None,
    context_hint: Optional[str] = None,
    alternate_questions: Optional[List[str]] = None,
) -> List[Dict]:
    """`query_embedding`: optional, from services.embeddings.get_embedding()
    on the raw user question -- SAME embedding the caller already computed
    for FAQ matching can be reused here (see faq.py), so this costs zero
    extra API calls in the common case.

    WHY THIS MATTERS FOR HYPOTHETICAL / UNALIASED QUESTIONS: everything
    else in this function (fulltext search, fuzzy title matching, the
    _FORCED_TOPIC_MATCHES/topic_aliases keyword patterns) is a *keyword*
    match -- it only finds a chunk if the question and the chunk share
    literal words. A hypothetical question about an offense nobody wrote
    an alias for ("what if I skip the flag ceremony") or a paraphrase
    that shares no vocabulary with the handbook's own wording will score
    near zero on all of that, even though the right chunk exists. Cosine
    similarity between the question's embedding and each chunk's own
    (precomputed, cached) embedding catches the MEANING match instead,
    with zero new code needed per topic/offense -- this is what lets the
    bot generalize past bullying/hazing to "any other type" of
    hypothetical without hand-writing a new alias for each one.
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        search_q = expand_query_for_retrieval(question)
        retrieval_queries = list(dict.fromkeys(
            query for query in (
                search_q,
                *(
                    expand_query_for_retrieval(variant)
                    for variant in (alternate_questions or [])
                ),
            )
            if query
        ))
        q_norm = normalize_input(question)
        q_words = set().union(*(_content_words(query) for query in retrieval_queries))

        rows_by_id: Dict[int, tuple] = {}
        for retrieval_query in retrieval_queries:
            try:
                cur.execute("""
                    SELECT id, chapter, section_title, page_start, page_end, content,
                           ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS relevance
                    FROM handbook_chunks
                    WHERE is_active = 1
                      AND search_vector @@ plainto_tsquery('english', %s)
                    ORDER BY relevance DESC
                    LIMIT %s
                """, (retrieval_query, retrieval_query, max_chunks * 8))
                for r in cur.fetchall():
                    hid = int(r[0])
                    if hid not in rows_by_id or float(r[6] or 0) > float(rows_by_id[hid][6] or 0):
                        rows_by_id[hid] = r
            except Exception as e:
                logger.warning(f"handbook fulltext query failed, falling back to fuzzy: {e}")
                conn.rollback()

        emb_by_id: Dict[int, Optional[list]] = {}
        try:
            cur.execute("""
                SELECT id, chapter, section_title, page_start, page_end, content, 0.0, embedding
                FROM handbook_chunks WHERE is_active = 1
            """)
            all_rows_raw = list(cur.fetchall())
            all_rows = [r[:7] for r in all_rows_raw]
            emb_by_id = {int(r[0]): r[7] for r in all_rows_raw}
        except Exception as e:
            logger.error(f"handbook title scan failed: {e}")
            all_rows = []

        title_hits = []
        for r in all_rows:
            best_ts = max(
                (_title_similarity(query, r[2] or "") for query in (question, *retrieval_queries)),
                default=0.0,
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

        forced_early = _forced_topic_for(question)
        if forced_early:
            title_needles_e, content_needles_e = forced_early
            for r in all_rows:
                title_l3 = (r[2] or '').lower()
                hid = int(r[0])
                if hid in rows_by_id:
                    continue
                title_hit3 = any(n in title_l3 for n in title_needles_e)
                if not title_hit3 and content_needles_e:
                    blob3 = f"{title_l3} {(r[5] or '')}".lower()
                    title_hit3 = any(n.lower() in blob3 for n in content_needles_e)
                if title_hit3:
                    rows_by_id[hid] = (r[0], r[1], r[2], r[3], r[4], r[5], 0.05)

        ai_picked_titles: List[str] = []
        ai_router_attempted = False
        try:
            all_titles = [r[2] for r in all_rows if r[2]]
            ai_router_attempted = bool(all_titles) and not circuit_breaker.is_open()
            router_context = context_hint
            if alternate_questions:
                alternate_hint = (
                    "Search paraphrases preserving the student's intended topic: "
                    + " | ".join(alternate_questions[:3])
                )
                router_context = "\n".join(
                    part for part in (context_hint, alternate_hint) if part
                )
            ai_picked_titles = llm_pick_sections(
                question, all_titles, max_picks=3, context_hint=router_context,
            )
        except Exception as e:
            logger.warning(f"AI section router skipped: {e}")
            ai_router_attempted = False
        if ai_picked_titles:
            ai_titles_l = {t.lower() for t in ai_picked_titles}
            for r in all_rows:
                hid = int(r[0])
                title_l4 = (r[2] or '').lower()
                if title_l4 in ai_titles_l and hid not in rows_by_id:
                    rows_by_id[hid] = (r[0], r[1], r[2], r[3], r[4], r[5], 0.05)

        if not rows_by_id:
            scored = []
            for r in all_rows:
                hid, chapter, title, ps, pe, content = r[0], r[1], r[2], r[3], r[4], r[5]
                blob = f"{title} {content[:600]}"
                score = max(
                    (
                        float(fuzzy_score(normalize_input(query), normalize_input(blob)))
                        for query in retrieval_queries
                    ),
                    default=float(fuzzy_score(q_norm, normalize_input(blob))),
                )
                chunk_embedding = emb_by_id.get(hid)
                if query_embedding and chunk_embedding is not None:
                    sem = max(0.0, cosine_similarity(query_embedding, chunk_embedding))
                    score = max(score, sem * 0.9)
                scored.append({
                    'id': hid, 'chapter': chapter, 'section_title': title,
                    'page_start': ps, 'page_end': pe, 'content': content,
                    'score': score, '_ai_router_attempted': ai_router_attempted,
                    '_raw_score': score,
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

            blob = f"{title_l} {content}".lower()

            title_sim = max(
                (_title_similarity(query, title) for query in (question, *retrieval_queries)),
                default=0.0,
            )

            kw_hits = sum(1 for w in (q_words - _GENERIC) if len(w) >= 4 and w in blob)
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

            q_l = question.lower()
            identity_q = any(w in q_l for w in (
                'logo', 'hymn', 'founded', 'founder', 'history', 'campus',
                'campuses', 'original name', 'full name', 'vision', 'mission',
                'philosophy', 'core values', 'demerit', 'offense', 'dress',
                'fraternity', 'suspension', 'expulsion', 'scholarship',
                'strand', 'senior high', 'bullying', 'appeal', 'dorm',
            ))
            admin_title = any(w in title_l for w in (
                'internship', 'requirements of the college dean', 'fees',
                'sipp', 'host training', 'memorandum of agreement',
            ))
            if identity_q and admin_title:
                noise *= 0.15

            distinctive = q_words - _GENERIC - _STOP
            dist_hits = sum(1 for w in distinctive if len(w) >= 4 and w in blob)
            dist_factor = 1.0 + 0.25 * min(dist_hits, 4) if dist_hits else 0.55

            # Prefer section titles that match the question topic
            title_boost = 1.0
            topic_title_cues = [
                (('strand', 'senior high', 'stem', 'humss', 'gas', 'abm', 'shs'),
                 ('program', 'academic', 'strand', 'offering')),
                (('program', 'course', 'courses', 'bachelor', 'what programs'),
                 ('program', 'academic', 'offering')),
                (('campus', 'campuses'),
                 ('program', 'history', 'campus', 'offering')),
                (('dress', 'uniform', 'attire', 'jewelry', 'haircut', 'appearance'),
                 ('dress', 'appearance', 'uniform')),
                (('dorm', 'dormitory', 'visit other'),
                 ('dorm', 'dormitory', 'residence', 'offense')),
                (('appeal', 'prescription'),
                 ('appeal', 'prescription', 'disciplinary')),
                (('bullying', 'anti-bullying'),
                 ('bullying', 'grievance', 'third')),
                (('safe spaces', '11313', 'harassment'),
                 ('safe spaces', 'gender', 'harassment', 'sexual')),
                (('scholarship', 'financial aid', 'financial assistance'),
                 ('scholarship', 'financial', 'admission', 'rights')),
                (('logo', 'emblem'), ('logo', 'symbol', 'emblem')),
                (('hymn',), ('hymn',)),
                (('founded', 'founder', 'history', 'original name'),
                 ('history', 'brief')),
                (('vision',), ('vision',)),
                (('mission',), ('mission',)),
                (('philosophy',), ('philosophy',)),
                (('core values',), ('core values', 'values')),
                (('demerit', 'first degree', 'second degree', 'third degree',
                  'first-degree', 'second-degree', 'third-degree', 'offense'),
                 ('degree', 'offense', 'demerit', 'behavioral')),
            ]
            for q_cues, t_cues in topic_title_cues:
                if any(c in q_l for c in q_cues):
                    if any(c in title_l for c in t_cues):
                        title_boost = max(title_boost, 2.2)
                    elif any(c in blob for c in q_cues):
                        title_boost = max(title_boost, 1.25)
                    else:
                        title_boost = min(title_boost, 0.7)
                    break

            # Degree-level questions must match the right offense section
            if re.search(r'\bsecond[ -]?degree\b', q_l):
                if 'second' in title_l and '15' in title_l:
                    title_boost = max(title_boost, 3.0)
                elif 'third' in title_l or 'first' in title_l:
                    title_boost *= 0.2
            if re.search(r'\bfirst[ -]?degree\b|\blight\s+offense', q_l):
                if 'first' in title_l and ('5' in title_l or 'demerit' in title_l):
                    title_boost = max(title_boost, 3.0)
                elif 'second' in title_l or 'third' in title_l:
                    title_boost *= 0.2
            if re.search(r'\bthird[ -]?degree\b|\bheaviest\b|\bserious\s+offense', q_l):
                if 'third' in title_l:
                    title_boost = max(title_boost, 3.0)
                elif 'first' in title_l or 'second' in title_l:
                    title_boost *= 0.2

            if re.search(r'\bdress\s*code|\buniform|\bjewelry|\bappearance\b', q_l):
                if 'dress' in title_l or 'appearance' in title_l or 'uniform' in title_l:
                    title_boost = max(title_boost, 3.0)

            if re.search(r'\bslippers?\b|\btsinelas\b', q_l):
                if 'first' in title_l and ('degree' in title_l or 'demerit' in title_l):
                    title_boost = max(title_boost, 4.0)
                elif 'dress' in title_l or 'appearance' in title_l:
                    title_boost *= 0.15
            if re.search(r'\bappeal\b', q_l):
                if 'appeal' in title_l or 'prescription' in title_l:
                    title_boost = max(title_boost, 3.0)
                elif 'sanction' in title_l and 'appeal' not in title_l:
                    title_boost *= 0.35
            if re.search(r'\b(safe spaces|11313)\b', q_l):
                if '11313' in blob or 'safe spaces' in title_l or 'gender' in title_l:
                    title_boost = max(title_boost, 3.0)
            if re.search(r'\b(anti.?bullying|10627)\b', q_l):
                if '10627' in blob or 'bullying' in blob:
                    title_boost = max(title_boost, 3.0)
            if re.search(r'\b(strand|senior high|stem|humss|gas|abm)\b', q_l):
                if 'program' in title_l or 'academic' in title_l or 'offering' in title_l:
                    title_boost = max(title_boost, 3.0)
                if 'scholarship' in title_l:
                    title_boost *= 0.15
                if 'senior citizen' in title_l or 'special needs' in title_l:
                    title_boost *= 0.08
            if re.search(r'\b(full name|real name|name of (this )?school)\b', q_l):
                if 'logo' in title_l or 'general provisions' in title_l or 'history' in title_l:
                    title_boost = max(title_boost, 2.5)
            if re.search(r'\b20\s*(or more )?demerits?|probation\b', q_l):
                if 'disciplinary measures' in title_l or 'probation' in blob:
                    title_boost = max(title_boost, 3.0)
            if re.search(r'\b(merit points?|start with|beginning of (the )?semester)\b', q_l):
                if 'behavioral code' in title_l or 'merit of 30' in blob:
                    title_boost = max(title_boost, 3.0)
                if 'third-degree' in title_l or 'third degree' in title_l:
                    title_boost *= 0.2

            conduct_intent = bool(re.search(
                r'\b(allowed|not allowed|prohibited|bawal|pwede\s*ba|okay to|'
                r'ok(?:ay)? to|can (we|i|students)|is it (ok|okay|allowed|fine)|'
                r'permitted|against the rules|violation)\b', q_l,
            ))
            if conduct_intent:
                distractor_titles = ('food services', 'dormitory charges', 'admission policies')
                if any(d in title_l for d in distractor_titles):
                    noise *= 0.12
                offense_or_grievance = ('offense', 'degree', 'grievance', 'behavioral')
                if (re.search(r'\b(food|cafeteria|eat|eating)\b', q_l)
                        and any(t in title_l for t in offense_or_grievance)):
                    title_boost = max(title_boost, 3.0)
                if (re.search(r'\bcook(ing)?\b', q_l) and 'dorm' in q_l
                        and any(t in title_l for t in offense_or_grievance)):
                    title_boost = max(title_boost, 3.0)
                if (re.search(r'\bvisit(ing)?\b', q_l) and 'dorm' in q_l
                        and any(t in title_l for t in offense_or_grievance)):
                    title_boost = max(title_boost, 3.0)

            if re.search(r'\b(max(imum)?\s*fine|fine for|biggest fine|money.{0,15}fine)\b', q_l):
                if ('disciplinary measures' in title_l or '1,500' in blob or '1500' in blob):
                    title_boost = max(title_boost, 3.5)

            if re.search(r'\b(want to become|aspire|future of|envision)\b', q_l):
                if 'vision' in title_l:
                    title_boost = max(title_boost, 4.0)
                elif 'mission' in title_l or 'philosophy' in title_l:
                    title_boost *= 0.4
            if re.search(r'\b(core values|pac values|the pac ones)\b', q_l):
                if 'core value' in title_l:
                    title_boost = max(title_boost, 4.0)
            if re.search(r'\b(heaviest|worst|most serious).{0,20}(demerit|level)\b', q_l):
                if 'third' in title_l:
                    title_boost = max(title_boost, 4.0)
                elif 'first' in title_l or 'second' in title_l:
                    title_boost *= 0.15
            if re.search(r'\b(how bad|how serious).{0,20}second\b', q_l):
                if 'second' in title_l:
                    title_boost = max(title_boost, 4.0)
                elif 'first' in title_l or 'third' in title_l:
                    title_boost *= 0.15
            if re.search(r'\b(hike|outing).{0,25}(chaperone|without)\b', q_l):
                if 'first' in title_l and 'degree' in title_l:
                    title_boost = max(title_boost, 4.0)
                elif any(t in title_l for t in ('second', 'third', 'charges', 'food service')):
                    title_boost *= 0.2
            if re.search(r'\b(visit|visiting).{0,20}(other )?dorms?\b', q_l):
                if 'first' in title_l and 'degree' in title_l:
                    title_boost = max(title_boost, 4.0)
                elif 'dormitory charges' in title_l or 'pledge' in title_l:
                    title_boost *= 0.15
            if re.search(r'\bcook(ing)?\b', q_l) and re.search(r'\b(dorm|dormitory)\b', q_l):
                if 'second' in title_l and 'degree' in title_l:
                    title_boost = max(title_boost, 4.0)
                elif 'dormitory charges' in title_l or 'first' in title_l:
                    title_boost *= 0.15
            if re.search(r'\b(food|cafeteria).{0,30}(not allowed|prohibited|bawal|banned)\b', q_l):
                if 'second' in title_l and 'degree' in title_l:
                    title_boost = max(title_boost, 4.0)
                elif 'food service' in title_l or ('cafeteria' in title_l and 'offense' not in title_l):
                    title_boost *= 0.12

            score = (
                (fts + 0.05)
                * (0.35 + 1.8 * title_sim)
                * length_factor
                * overlap_factor
                * noise
                * dist_factor
                * title_boost
            )
            scored.append({
                'id': r[0], 'chapter': r[1], 'section_title': r[2],
                'page_start': r[3], 'page_end': r[4], 'content': content,
                'score': score, '_ai_router_attempted': ai_router_attempted,
            })

        forced = _forced_topic_for(question)
        if forced and scored:
            title_needles, content_needles = forced
            for s in scored:
                title_l2 = (s.get('section_title') or '').lower()
                blob2 = f"{title_l2} {(s.get('content') or '')}".lower()
                title_hit = any(n in title_l2 for n in title_needles)
                content_hit = any(n.lower() in blob2 for n in content_needles) if content_needles else False
                if title_hit:
                    s['score'] = float(s['score']) * 20.0
                    s['_forced'] = True
                elif content_hit:
                    s['score'] = float(s['score']) * 3.0
                    s['_forced_content'] = True

        if ai_picked_titles:
            ai_titles_l2 = {t.lower() for t in ai_picked_titles}
            for s in scored:
                if (s.get('section_title') or '').lower() in ai_titles_l2:
                    s['score'] = float(s.get('score') or 0) * 30.0
                    s['_ai_picked'] = True

        try:
            idf = get_idf_index(_load_all_chunk_texts)
            scored = rarity_rerank(
                question, scored, idf,
                text_fn=lambda c: f"{c.get('section_title', '')} {c.get('content', '')}",
                score_key='score', blend=0.12,
            )
        except Exception as e:
            logger.warning(f"rarity_rerank skipped: {e}")


        if query_embedding:
            for s in scored:
                chunk_id = s.get('id')
                if not isinstance(chunk_id, int):
                    continue
                chunk_emb = emb_by_id.get(chunk_id)
                if not chunk_emb:
                    continue
                sem = max(0.0, cosine_similarity(query_embedding, chunk_emb))
                if sem >= 0.72:
                    s['score'] = max(float(s.get('score') or 0), sem)
                    s['_semantic_score'] = sem

        # Re-apply forced boost after rarity_rerank
        if forced and scored:
            title_needles, content_needles = forced
            for s in scored:
                title_l2 = (s.get('section_title') or '').lower()
                blob2 = f"{title_l2} {(s.get('content') or '')}".lower()
                title_hit = any(n in title_l2 for n in title_needles)
                content_hit = any(n.lower() in blob2 for n in content_needles) if content_needles else False
                if title_hit:
                    s['score'] = float(s.get('score') or 0) * 15.0
                    s['_forced'] = True
                elif content_hit:
                    s['score'] = float(s.get('score') or 0) * 2.0
                    s['_forced_content'] = True


        if ai_picked_titles:
            ai_titles_l3 = {t.lower() for t in ai_picked_titles}
            for s in scored:
                if (s.get('section_title') or '').lower() in ai_titles_l3:
                    s['score'] = float(s.get('score') or 0) * 20.0
                    s['_ai_picked'] = True

        scored.sort(key=lambda x: x['score'], reverse=True)
        top = float(scored[0]['score'] or 1.0)
        for s in scored:
            s['_raw_score'] = float(s.get('score') or 0)
            s['score'] = float(s['score']) / top if top else 0.0
        return scored[:max_chunks]

    except Exception as e:
        logger.error(f"get_top_handbook_chunks error: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def get_top_handbook_chunks(
    question: str,
    max_chunks: int = 8,
    query_embedding: Optional[List[float]] = None,
    context_hint: Optional[str] = None,
    detected_lang: str = "english",
) -> List[Dict]:
    """Search the original and canonical paraphrases for Filipino questions."""
    variants = llm_expand_school_query(question, detected_lang)
    chunks = _get_top_handbook_chunks(
        question,
        max_chunks=max_chunks,
        query_embedding=query_embedding,
        context_hint=context_hint,
        alternate_questions=variants or None,
    )
    return chunks


def get_exact_handbook_answer(question: str) -> Optional[Tuple[str, List[str]]]:
    """Return current handbook source text for a question previously shown as a prompt.

    Landing-page popular handbook questions retain their matched section titles
    in chat history. Exact-click reuse can serve those current source excerpts
    without rerunning the AI router or answer generator.
    """
    if not question:
        return None

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT hc.section_title, hc.content
            FROM chat_history ch
            JOIN handbook_chunks hc
              ON hc.section_title = ANY(string_to_array(ch.source_ref, '; '))
            WHERE ch.user_message = %s
              AND ch.category = 'handbook_query'
              AND ch.source_ref IS NOT NULL
              AND hc.is_active = 1
            ORDER BY hc.section_title
            LIMIT 3
            """,
            (question,),
        )
        rows = cur.fetchall()
        rows = [
            (title, content)
            for title, content in rows
            if title and isinstance(content, str) and content.strip()
        ]
        if not rows:
            return None
        return "\n\n".join(content.strip() for _, content in rows), [
            title for title, _ in rows
        ]
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
    if chunk.get('_ai_picked'):
        return True
    if chunk.get('_forced') or chunk.get('_forced_content'):
        return True

    if chunk.get('_semantic_score', 0.0) >= 0.80:
        return True

    hits, required = _content_word_overlap_score(question, chunk)
    title_score = _title_match_score(question, chunk)
    expanded = expand_query_for_retrieval(question)
    title_score = max(
        title_score,
        _title_similarity(expanded, chunk.get('section_title') or ''),
    )
    lead_score = _lead_fuzzy_score(question, chunk)

    exp_words = _content_words(expanded) - _GENERIC
    title = (chunk.get('section_title') or '')
    content = chunk.get('content') or ''
    # Full content so deep list facts still count
    blob = f"{title} {content}".lower()
    exp_hits = sum(1 for w in exp_words if w in blob)

    forced = _forced_topic_for(question)
    if forced:
        title_needles, content_needles = forced
        title_l_chk = title.lower()
        if any(n in title_l_chk for n in title_needles):
            return True
        if content_needles and any(n.lower() in blob for n in content_needles):
            return True

    q_l = (question or '').lower()
    title_l = title.lower()
    identity_q = any(w in q_l for w in (
        'logo', 'hymn', 'founded', 'founder', 'history', 'campus', 'campuses',
        'original name', 'full name', 'vision', 'philosophy', 'core values',
        'demerit', 'offense', 'dress', 'fraternity', 'bullying', 'safe spaces',
    ))
    admin_title = any(w in title_l for w in (
        'internship', 'requirements of the college dean', 'fees',
        'host training', 'sipp coordinator',
    ))
    if identity_q and admin_title and title_score < 0.55:
        return False

    if title_score >= 0.40:
        return True
    if title_score >= 0.28 and (hits >= 1 or exp_hits >= 2):
        return True
    if exp_hits >= 2:
        return True
    if hits >= required and (title_score >= 0.25 or lead_score >= 0.30):
        return True
    if hits >= required + 1 or exp_hits >= required + 1:
        return True
    return False


def _trim_to_close_contenders(chunks: List[Dict], score_key: str = 'score',
                               max_gap: float = 0.12, max_n: int = 4) -> List[Dict]:
    """Cross-contamination guard: `hard`/`semantic_ok` used to hand the LLM
    up to 4 chunks just because each individually cleared a bar, even when
    they're about unrelated offenses/topics that happen to share vocabulary
    or land in similar embedding space (e.g. a broad "what's the penalty
    for X" pulling in both the section that actually covers X AND a
    same-severity-tier section listing a completely different offense).
    _build_handbook_answer_from_chunks concatenates every returned chunk
    into ONE prompt, so the more chunks go in, the more surface area there
    is for the model to blend facts across them. This keeps the leader
    plus only genuine close contenders (within max_gap of the top score),
    same "don't ask about a mediocre #2 just because it's non-zero"
    principle as services.disambiguation -- not the same use case, but the
    same shape of bug."""
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda c: c.get(score_key, 0.0), reverse=True)
    top_score = ordered[0].get(score_key, 0.0)
    out = []
    for c in ordered:
        if (top_score - c.get(score_key, 0.0)) > max_gap:
            break
        out.append(c)
        if len(out) >= max_n:
            break
    return out


def get_relevant_chunks(question: str, candidates: List[Dict]) -> List[Dict]:
    if not candidates:
        return []
    ai_picked = [c for c in candidates if c.get('_ai_picked')]
    forced_title = [c for c in candidates if c.get('_forced')]
    if forced_title and ai_picked:
        merged = forced_title + [c for c in ai_picked if c not in forced_title]
        return _trim_to_close_contenders(merged, max_n=3)
    if ai_picked:
        return _trim_to_close_contenders(ai_picked, max_n=3)
    if forced_title:
        return forced_title[:3]
    forced_content = [c for c in candidates if c.get('_forced_content')]
    if forced_content:
        return forced_content[:2]
    hard = [c for c in candidates if chunk_is_relevant_hard(question, c)]
    if hard:
        return _trim_to_close_contenders(hard, max_n=4)
    semantic_ok = [c for c in candidates if c.get('_semantic_score', 0.0) >= 0.68]
    if semantic_ok:
        return _trim_to_close_contenders(semantic_ok, score_key='_semantic_score',
                                          max_gap=0.06, max_n=2)
    llm_kept = []
    for c in candidates[:4]:
        if llm_hard_relevance(question, c):
            llm_kept.append(c)
    if llm_kept:
        return llm_kept[:4]

    router_had_a_say = any(c.get('_ai_router_attempted') for c in candidates)
    if router_had_a_say:
        return []
    if candidates and candidates[0].get('score', 0) >= 0.45:
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
    chosen = [s for hits, s in ranked[:max_sentences] if hits > 0]
    if not chosen:
        chosen = [s for _, s in ranked[:2]]
    return ' '.join(chosen)


def _answer_relevant_to_question(question: str, answer: str, min_overlap: int = 1) -> bool:
    q_words = _content_words(expand_query_for_retrieval(question)) - _GENERIC
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

            lang_instruction = (
                f"Reply in {nlg.lang_name(detected_lang)}." if detected_lang != 'english' else ""
            )
            few_shot = nlg.few_shot_example(detected_lang) if detected_lang != 'english' else ""
            casual_note = nlg.casual_tone_note(question)
            prompt = f"""{nlg.now_context()}

            You are a friendly student helpdesk chatbot. Answer the student's question using ONLY the handbook excerpts below.
            Do not invent any facts, names, numbers, or rules that are not present in the excerpts.
            If the excerpts do not contain enough information to answer the question, reply with exactly: CANNOT_ANSWER
            {"If more than one excerpt is given below and they cover DIFFERENT topics/offenses, use only the excerpt that actually answers the student's question -- never combine a fact, number, or penalty from one excerpt with a fact from a different excerpt about something else." if len(context_parts) > 1 else ""}

            Student question: {question}

            Handbook excerpts:
            {context}

            Instructions:
            - Sound natural and clear — like a real chatbot, not a formal document.
            - Identify the exact fact requested and answer that directly first; do not substitute related facts from the same excerpt.
            - NEVER start with "According to the section…" or similar citations. Just answer.
            - If the question is yes/no (starts with Is/Are/Can/Do/Does/Will/May/Allowed/
              Pwede/Okay ba), your FIRST WORD must be "Yes" or "No", then a short reason.
            - If the student named a SPECIFIC item (e.g. "shorts", "earrings") and the
              excerpts only cover a DIFFERENT, similar item (e.g. jeans, jewelry in
              general) without actually naming the one asked about, do not answer as if
              the specific item was covered — say plainly that the excerpt doesn't
              specifically address it, or reply CANNOT_ANSWER if nothing else applies.
            - Format for easy reading on EVERY answer: use **bold** for key terms, line breaks between points, blank lines between sections. For lists put one item per line with - . For lyrics/hymns put each line on its own line and **bold** labels like **Chorus:**.
            - State numbers (demerits, days, fines) clearly.
            - Optional: a light page note at the end only, e.g. (More detail is on page 7.)
            - {nlg.STYLE_GUIDE}
            - {lang_instruction}
            {few_shot}{casual_note}
            When the student asked in Bisaya or Tagalog, reply fully in that language (not English) --
            translate/paraphrase the excerpt content into it, don't just answer in English.

            Answer:"""

            response = llm_chat(prompt, max_tokens=450)
            if response and "CANNOT_ANSWER" not in response.upper():
                response = response.strip()
                is_forced = bool(_forced_topic_for(question)) or any(
                    c.get('_forced') or c.get('_forced_content') for c in chunks
                )
                min_ov = 0.25 if is_forced else 0.35
                # Translated answers need semantic validation, not English word overlap.
                if (
                    detected_lang == 'english'
                    and not answer_is_grounded(response, source_blob, min_overlap=min_ov)
                ):
                    logger.info(f"handbook grounding rejected for q={question[:80]!r}")
                    response = None
                elif (
                    detected_lang == 'english'
                    and not _answer_relevant_to_question(question, response, min_overlap=1)
                ):
                    logger.info(f"handbook answer-relevance REJECTED (is_forced={is_forced}) "
                                f"for q={question[:80]!r}")
                    response = None
                elif not verify_output_language(response, detected_lang):
                    logger.info(
                        f"handbook answer drifted language (wanted {detected_lang}), "
                        f"retrying once for q={question[:80]!r}"
                    )
                    retry_prompt = prompt + (
                        f"\n\nYour previous attempt was not in {nlg.lang_name(detected_lang)} — "
                        f"try again, and make sure the reply is actually in "
                        f"{nlg.lang_name(detected_lang)} this time, not English."
                    )
                    try:
                        retry_response = llm_chat(retry_prompt, max_tokens=480)
                    except Exception:
                        retry_response = None
                    retry_ok = (
                        retry_response and retry_response.strip()
                        and "CANNOT_ANSWER" not in retry_response.upper()
                        and (
                            detected_lang != 'english'
                            or answer_is_grounded(
                                retry_response.strip(), source_blob, min_overlap=min_ov
                            )
                        )
                        and verify_output_language(retry_response.strip(), detected_lang)
                    )
                    if retry_ok and retry_response is not None:
                        response = retry_response.strip()
                    else:
                        logger.info(
                            "handbook answer retry still didn't land in the right language; "
                            "falling through to the deterministic excerpt-extraction path below."
                        )
                        response = None
                if response and "CANNOT_ANSWER" not in response.upper():
                    _NEGATION_MARKERS = (
                        "doesn't mention", "does not mention", "don't mention",
                        "not mentioned", "no mention", "doesn't contain",
                        "does not contain", "no reference", "isn't mentioned",
                        "is not mentioned", "doesn't specifically address",
                        "does not specifically address", "doesn't cover",
                        "does not cover", "not covered", "doesn't say",
                        "does not say",
                    )
                    _is_negation = any(m in response.lower() for m in _NEGATION_MARKERS)
                    if len(chunks) > 1 and not _is_negation and detected_lang == 'english':
                        
                        top_only_ratio = grounding_ratio(response, chunks[0].get('content', ''))
                        if top_only_ratio < 0.15:
                            logger.info(
                                f"handbook answer for q={question[:80]!r} grounds mostly in a "
                                f"NON-top excerpt ({len(chunks)} excerpts offered, top-excerpt "
                                f"overlap={top_only_ratio:.2f}) -- rejecting, falling through to "
                                f"single-excerpt extraction."
                            )
                            response = None
                if response and "CANNOT_ANSWER" not in response.upper():
                    if llm_self_check(question, response, source_blob):
                        return response
                    logger.info(f"handbook answer self-check REJECTED for q={question[:80]!r}")
        except Exception as e:
            logger.error(f"AI handbook answer failed: {e}")

    forced = _forced_topic_for(question)
    top = chunks[0]
    title = top.get('section_title', 'the Student Handbook')
    page = top.get('page_start')
    content = (top.get('content') or '').strip()

    clean = _extract_relevant_sentences(content, question, max_sentences=6)
    if not clean and forced:
        _, content_needles = forced
        lines = []
        for line in content.split('\n'):
            low = line.lower()
            if any(n.lower() in low for n in (content_needles or [])):
                lines.append(line.strip())
        if lines:
            clean = ' '.join(lines[:5])
    if not clean:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        clean = ' '.join(s.strip() for s in sentences[:3] if len(s.strip()) > 25)

    if forced and content:
        if not clean:
            clean = content[:600].strip()
        if (
            detected_lang == 'english'
            and not _answer_relevant_to_question(question, content)
        ):
            logger.info(
                f"handbook forced-topic fallback REJECTED (chunk irrelevant to question) "
                f"for q={question[:80]!r} chunk={title!r}"
            )
            return None
        page_note = f"\n\n(More detail is on page {page}.)" if page else ""
        final = f"{clean}{page_note}".strip()
        final = nlg.speak_naturally(final, question, detected_lang=detected_lang)
        return final

    if not clean or (
        detected_lang == 'english'
        and not _answer_relevant_to_question(question, clean)
    ):
        return None
    page_note = f"\n\n(More detail is on page {page}.)" if page else ""
    final = f"{clean}{page_note}".strip()
    final = nlg.speak_naturally(final, question, detected_lang=detected_lang)
    return final


def _merge_compound_answers(original_question: str, sub_answers: List[Tuple[str, str]],
                            detected_lang: str = 'english') -> Optional[str]:
    """Smooth two-or-more already-correct, already-grounded sub-answers into
    one natural reply, the same "paraphrase, never compose" pattern used by
    nlg.speak_naturally — the model is only allowed to reorganize/reword,
    never add a fact that isn't already in one of the sub-answers. Falls
    back to None (caller then just concatenates the raw sub-answers) if the
    AI is unavailable or its merge doesn't check out.
    """
    if circuit_breaker.is_open():
        return None

    canonical = "\n\n".join(a for _, a in sub_answers)
    casual_note = nlg.casual_tone_note(original_question)
    prompt = f"""{nlg.now_context()}

A student asked a question with two parts: "{original_question}"

Here are the factually-correct answers to each part, already grounded in the handbook:

{canonical}

Combine these into a single natural reply that clearly addresses BOTH parts, in {nlg.lang_name(detected_lang)}. {nlg.STYLE_GUIDE}{casual_note}

STRICT RULE: Do not add, remove, or change any fact from the answers above — no dates,
names, numbers, or rules beyond exactly what's already stated. You may only reorganize
and smooth the wording so it reads as one reply, never invent new content. If you're not
confident you can do that safely, just return the two answers above joined with a line
break, unchanged.

Combined answer:"""

    try:
        response = llm_chat(prompt, max_tokens=500)
        if not response:
            return None
        response = response.strip()
        if not answer_is_grounded(response, canonical, min_overlap=0.35):
            logger.info(f"compound handbook merge failed grounding check for "
                        f"q={original_question[:80]!r}, falling back to concatenation")
            return None
        if not verify_output_language(response, detected_lang):
            logger.info(f"compound handbook merge drifted language for "
                        f"q={original_question[:80]!r}, falling back to concatenation")
            return None
        return response
    except Exception as e:
        logger.error(f"_merge_compound_answers error: {e}")
        return None


def _answer_compound_handbook_question(original_question: str, parts: List[str],
                                       detected_lang: str = 'english',
                                       min_score_to_answer: Optional[float] = None
                                       ) -> Optional[str]:
    """Answer each part of a compound question (e.g. "what's the school's
    real name and where are the campuses") independently, through the exact
    same single-topic retrieval + grounding pipeline used for an ordinary
    question, then combine the results into one reply.

    This exists because get_top_handbook_chunks()/chunk_is_relevant_hard()
    are tuned (title-boost heuristics, identity-vs-admin-chapter rejection,
    etc.) around ONE dominant topic per query — asking them to serve two
    unrelated topics in a single query search tends to retrieve chunks for
    only one half (or, worse, neither), which is why compound questions
    like Q1 in the test report ("real full name ... and where are the
    campuses?") fell through to CANNOT_ANSWER even though both individual
    facts are answerable on their own. Running retrieval once per part
    sidesteps that entirely — nothing about the retrieval/grounding logic
    itself changes.

    Disambiguation is intentionally skipped on this path (each part is
    usually specific enough not to need it, and picking a disambiguation
    option for half of a two-part question doesn't have anywhere clean to
    go); a part whose top chunk is ambiguous will just fall back to normal
    single-chunk answering via _build_handbook_answer_from_chunks.
    """
    sub_answers: List[Tuple[str, str]] = []
    for part in parts:
        chunks = get_top_handbook_chunks(part)
        if not chunks:
            continue
        if min_score_to_answer is not None:
            top_score = chunks[0].get('score', 0)
            if 0 <= top_score <= 1 and top_score < min_score_to_answer:
                continue
        answer = _build_handbook_answer_from_chunks(part, chunks[:5], detected_lang)
        if answer:
            sub_answers.append((part, answer))

    if not sub_answers:
        return None

    # Only one half turned out to be answerable — answer that half rather
    # than discarding a perfectly good answer because its sibling wasn't
    # found (same "answer what you honestly can" principle used elsewhere
    # in this module, e.g. the single-chunk fallback in
    # _build_handbook_answer_from_chunks).
    if len(sub_answers) == 1:
        return sub_answers[0][1]

    merged = _merge_compound_answers(original_question, sub_answers, detected_lang)
    if merged:
        return merged
    return "\n\n".join(a for _, a in sub_answers)


def answer_handbook_question(question: str, detected_lang: str = 'english',
                             min_score_to_answer: Optional[float] = None,
                             ctx=None) -> Optional[str]:
    context_hint = None
    if ctx is not None:
        hint_parts = []
        if getattr(ctx, 'last_handbook_sections', None):
            hint_parts.append(
                "Previously discussed section(s): "
                + ", ".join(ctx.last_handbook_sections)
            )
        history = getattr(ctx, 'conversation_history', None) or []
        if history and history[-1] is not None:
            previous_exchange = history[-1]
            hint_parts.append(f"Previous student question: {previous_exchange.get('user', '')}")
            hint_parts.append(f"Previous answer given: {(previous_exchange.get('bot', '') or '')[:300]}")
        if hint_parts:
            context_hint = "\n".join(hint_parts)

    cache_key = f"handbook_{hash(question.lower().strip())}_{hash(context_hint or '')}"
    cached = ai_cache.get(cache_key)
    if cached:
        return cached

    parts = split_compound_question(question)
    if len(parts) >= 2:
        response = _answer_compound_handbook_question(
            question, parts, detected_lang, min_score_to_answer)
        if response:
            ai_cache.set(cache_key, response)
            if ctx is not None:
                ctx.clear_disambiguation()
        return response

    chunks = get_top_handbook_chunks(
        question,
        context_hint=context_hint,
        detected_lang=detected_lang,
    )
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
            used_titles = [c.get('section_title') for c in chunks[:3] if c.get('section_title')]
            ctx.update(handbook_sections=used_titles or None, clear_schedule_context=True)
    return response


def answer_from_selected_chunk(original_question: str, chunk: Dict,
                               detected_lang: str = 'english') -> Optional[str]:
    return _build_handbook_answer_from_chunks(original_question, [chunk], detected_lang)