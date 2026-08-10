"""
Generic Student Handbook PDF / JSON ingestion + FAQ-pair generation.

- PDF → sections (heading split, list-aware, noise filter)
- Optional LLM refine (third-party)
- Optional conversion of sections → FAQ-style Q+A pairs
- Mass insert of pasted / generated FAQ pairs into the faq table
- No hardcoded institutional facts
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import pdfplumber

logger = logging.getLogger("pac_helpdesk.handbook_ingestion")

HAS_AI = False
call_for_ingest = None
get_groq_usage_report = None
GROQ_API_KEYS: List[str] = []

try:
    from ai_providers import call_for_ingest, get_groq_usage_report, GROQ_API_KEYS
    HAS_AI = True
except Exception:
    try:
        from ai_providers.providers import call_for_ingest, get_groq_usage_report, GROQ_API_KEYS
        HAS_AI = True
    except Exception as e:
        logger.error("AI providers import failed: %s", e)
        HAS_AI = False


MAX_CHUNK_CHARS = 2800
MAX_CHUNK_CHARS_LIST = 9000
OVERLAP_CHARS = 280
MIN_CHUNK_CHARS = 80

MAX_WORKERS = 1
LLM_MAX_TOKENS = 2200
PER_CALL_PAUSE = 1.8
BATCH_COOLDOWN_EVERY = 25
BATCH_COOLDOWN_SECONDS = 12

_PART_RX = re.compile(r'^(PART|ARTICLE|CHAPTER|SECTION)\s+[IVXLCDM\d]+', re.I)
_LETTER_RX = re.compile(r'^[A-Z]\.\s+[A-Z]')
_ROMAN_RX = re.compile(r'^[IVXLCDM]{1,5}\.\s+[A-Z]')
_NUMBERED_HEADING_RX = re.compile(r'^\d{1,2}(\.\d{1,2})*\.?\s+[A-Z]')
_STANDALONE = {
    'philosophy', 'vision', 'mission', 'goals', 'core values',
    'foreword', 'preamble', 'introduction', 'preface',
}

LIST_ITEM_RX = re.compile(
    r'^\s*(?:\d{1,2}|[a-zA-Z]|[ivxlcdm]{1,5})[.)]\s+\S',
    re.MULTILINE
)
_NUMERIC_START = re.compile(r'^\d{1,3}[.)]\s+\S')


def _looks_like_list_item(line: str) -> bool:
    if not _NUMERIC_START.match(line):
        return False
    words = line.split()
    if line.rstrip().endswith(('.', ';', ':')):
        return True
    if len(words) <= 10:
        return True
    return False


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 160:
        return False
    if _looks_like_list_item(line):
        return False
    words = line.split()
    if not words or len(words) > 18:
        return False
    if _PART_RX.match(line):
        return True
    if _LETTER_RX.match(line):
        return True
    if _ROMAN_RX.match(line):
        return True
    if _NUMBERED_HEADING_RX.match(line) and len(words) <= 12:
        return True
    if line.lower().rstrip('.') in _STANDALONE:
        return True
    letters = [c for c in line if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio >= 0.78 and len(words) <= 14:
            return True
    if re.search(r'(FIRST|SECOND|THIRD).{0,12}(DEGREE|LEVEL).{0,12}(OFFENSE|OFFENCE)', line, re.I):
        return True
    if re.search(r'\b(DRESS|APPEARANCE|UNIFORM|JEWELRY|LOGO|EMBLEM|HYMN|HISTORY)\b', line, re.I) and len(words) <= 10:
        return True
    return False


def _is_list_heavy(content: str) -> bool:
    return len(LIST_ITEM_RX.findall(content)) >= 3


def _is_offense_list(title: str, content: str) -> bool:
    t = (title + " " + content[:400]).lower()
    return bool(re.search(
        r'(first|second|third).{0,20}(degree|level).{0,20}(offense|offence|demerit)',
        t
    ))


def _norm_title(t: str) -> str:
    t = (t or '').lower()
    t = re.sub(r'\(part\s*\d+\)', '', t)
    t = re.sub(r'[^a-z0-9]+', ' ', t).strip()
    return t


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for table in (page.extract_tables() or []):
                for row in table:
                    cells = [str(c).strip() for c in row if c and str(c).strip()]
                    if cells:
                        text += "\n" + " | ".join(cells)
            if not text.strip():
                logger.warning("Page %s: no extractable text", i)
            pages.append((i, text))
    return pages


def _find_break_point(content: str, target: int, list_heavy: bool = False) -> int:
    if target >= len(content):
        return len(content)
    window_start = max(0, target - 450)
    window = content[window_start:target]
    if list_heavy:
        idx = window.rfind('\n\n')
        if idx != -1:
            return window_start + idx + 2
        last = None
        for m in LIST_ITEM_RX.finditer(content, window_start, target + 250):
            last = m
        if last:
            return last.start()
    for sep in ('\n\n', '.\n', '. ', '\n', '; ', ' '):
        idx = window.rfind(sep)
        if idx != -1:
            return window_start + idx + len(sep)
    return target


def split_into_sections(pages: List[Tuple[int, str]]) -> List[Dict]:
    sections = []
    current = None
    for page_num, text in pages:
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                if current is not None:
                    current['content'] += '\n'
                continue
            if _is_heading(stripped):
                if current and current['content'].strip():
                    sections.append(current)
                current = {
                    'chapter': stripped,
                    'section_title': stripped,
                    'page_start': page_num,
                    'page_end': page_num,
                    'content': '',
                }
            else:
                if current is None:
                    current = {
                        'chapter': 'General',
                        'section_title': 'Introduction',
                        'page_start': page_num,
                        'page_end': page_num,
                        'content': '',
                    }
                current['content'] += line + '\n'
                current['page_end'] = page_num
    if current and current['content'].strip():
        sections.append(current)

    final = []
    for sec in sections:
        content = re.sub(r'\n{3,}', '\n\n', sec['content']).strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        title = sec['section_title']
        up = title.upper()
        # Light canonicalization only — still derived from the text itself
        if re.search(r'BEHAVIORAL\s+CODE|CODE\s+OF\s+CONDUCT', up):
            title = "Student's Behavioral Code"
        elif re.search(r'FIRST.?DEGREE|FIRST\s+DEGREE', up):
            title = "First-Degree Offense (5 Demerits)"
        elif re.search(r'SECOND.?DEGREE|SECOND\s+DEGREE', up):
            title = "Second-Degree Offense (15 Demerits)"
        elif re.search(r'THIRD.?DEGREE|THIRD\s+DEGREE', up):
            title = "Third-Degree Offense (30 Demerits)"
        elif re.search(r'DRESS\s+(AND\s+)?APPEARANCE|DRESS\s+CODE', up):
            title = "Dress and Appearance"
        elif re.search(r'\bVISION\b', up) and 'MISSION' not in up:
            title = "Vision"
        elif re.search(r'\bMISSION\b', up) and 'VISION' not in up:
            title = "Mission"
        elif re.search(r'CORE\s+VALUES', up):
            title = "Core Values"
        elif re.search(r'\bPHILOSOPHY\b', up):
            title = "Philosophy"
        elif re.search(r'^GOALS?$', up):
            title = "Goals"
        elif re.search(r'BRIEF\s+HISTORY|HISTORY\s+OF', up):
            title = "Brief History of PAC"
        elif re.search(r'\bHYMN\b', up):
            title = "PAC Hymn"
        elif re.search(r'LOGO|EMBLEM|SYMBOLS?', up):
            title = "Logo and Symbols"
        sec['section_title'] = title

        list_heavy = _is_list_heavy(content) or _is_offense_list(title, content)
        cap = MAX_CHUNK_CHARS_LIST if list_heavy else MAX_CHUNK_CHARS

        if _is_offense_list(title, content):
            final.append({**sec, 'content': content})
            continue
        if len(content) <= cap:
            final.append({**sec, 'content': content})
            continue

        start, part = 0, 1
        while start < len(content):
            target_end = min(start + cap, len(content))
            end = _find_break_point(content, target_end, list_heavy=list_heavy)
            if end <= start:
                end = target_end
            piece = content[start:end].strip()
            if len(piece) >= MIN_CHUNK_CHARS:
                chunk_title = title if part == 1 else f"{title} (part {part})"
                final.append({
                    'chapter': sec['chapter'],
                    'section_title': chunk_title,
                    'page_start': sec['page_start'],
                    'page_end': sec['page_end'],
                    'content': piece,
                })
                part += 1
            if end >= len(content):
                break
            overlap_target = max(start, end - OVERLAP_CHARS)
            start = _find_break_point(content, overlap_target, list_heavy=list_heavy)
            if start >= end:
                start = end
    return final


def is_noise_chunk(ch: Dict) -> bool:
    title = (ch.get('section_title') or '').lower()
    content = (ch.get('content') or '').strip()
    if len(content) < MIN_CHUNK_CHARS:
        return True
    if 'table of contents' in title or title in ('introduction', 'general'):
        if len(content) < 400:
            return True
    if re.fullmatch(r'[\d\s\.…\-–—]+', content):
        return True
    cleaned = re.sub(r'[\d\W]+', '', content)
    if len(cleaned) < 40:
        return True
    return False


def fuzzy_overlap(a: str, b: str) -> float:
    sa = set(re.findall(r'\w{4,}', a.lower()))
    sb = set(re.findall(r'\w{4,}', b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    by_title: Dict[str, List[int]] = {}
    for i, ch in enumerate(chunks):
        by_title.setdefault(_norm_title(ch['section_title']), []).append(i)
    drop = set()
    for idxs in by_title.values():
        if len(idxs) < 2:
            continue
        idxs_sorted = sorted(
            idxs,
            key=lambda i: (chunks[i].get('priority', 0), len(chunks[i].get('content', ''))),
            reverse=True
        )
        keeper = chunks[idxs_sorted[0]]
        for i in idxs_sorted[1:]:
            other = chunks[i]
            a = (other.get('content') or '')[:300]
            b = (keeper.get('content') or '')
            if a and a in b:
                drop.add(i)
            elif len(a) > 100 and fuzzy_overlap(a, b) > 0.72:
                drop.add(i)
    return [ch for i, ch in enumerate(chunks) if i not in drop]


_DOMAIN_KEEP = {
    'dress', 'code', 'uniform', 'attire', 'appearance', 'jewelry', 'jewellery',
    'attendance', 'absent', 'absence', 'tardy', 'conduct', 'discipline',
    'disciplinary', 'suspension', 'expulsion', 'violation', 'penalty', 'sanction',
    'offense', 'offence', 'merit', 'demerit', 'points', 'curfew', 'dormitory',
    'fraternity', 'sorority', 'hazing', 'bullying', 'philosophy', 'vision',
    'mission', 'values', 'core', 'logo', 'emblem', 'symbol', 'hymn', 'motto',
    'campus', 'founded', 'founder', 'history', 'admission', 'probation',
    'necklace', 'bracelet', 'ring', 'haircut',
}


def build_keywords(content: str, section_title: str, max_kw: int = 22) -> str:
    text = (section_title + ' ' + content).lower()
    words = re.findall(r'\b[a-zA-Z0-9][a-zA-Z0-9\-]{2,}\b', text)
    stop = {
        'shall', 'this', 'that', 'with', 'from', 'have', 'will', 'must', 'each',
        'their', 'they', 'them', 'been', 'were', 'being', 'which', 'where',
        'when', 'what', 'into', 'upon', 'such', 'also', 'than', 'only', 'other',
        'about', 'after', 'before', 'under', 'over', 'more', 'most', 'some',
        'any', 'all', 'may', 'can', 'should', 'would', 'could', 'the', 'and',
        'for', 'are', 'not', 'you', 'your', 'our', 'his', 'her', 'its',
    }
    freq = {}
    for w in words:
        if w in stop and w not in _DOMAIN_KEEP:
            continue
        if len(w) < 4 and w not in _DOMAIN_KEEP and not w.isdigit():
            continue
        freq[w] = freq.get(w, 0) + 1
    for w in re.findall(r'\b[a-zA-Z0-9][a-zA-Z0-9\-]{2,}\b', section_title.lower()):
        if w not in stop or w in _DOMAIN_KEEP:
            freq[w] = freq.get(w, 0) + 6
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_kw]
    return ', '.join(w for w, _ in top)


def enrich_keywords(title: str, content: str, existing: str = '') -> str:
    t = (title + ' ' + content[:800]).lower()
    extra = []
    if 'vision' in t or 'envisions' in t:
        extra += ['vision', 'center of excellence']
    if 'mission' in t and 'commission' not in t:
        extra += ['mission', 'human service', 'personal integrity']
    if 'philosophy' in t:
        extra += ['philosophy', 'god is the source', 'true education']
    if 'core value' in t or 'purpose-driven' in t:
        extra += ['core values', 'purpose-driven', 'christ-centered']
    if re.search(r'first.{0,10}degree|5\s+demerit', t):
        extra += ['first degree', '5 demerits', 'first-degree offense']
    if re.search(r'second.{0,10}degree|15\s+demerit', t):
        extra += ['second degree', '15 demerits', 'second-degree offense']
    if re.search(r'third.{0,10}degree|30\s+demerit', t):
        extra += ['third degree', '30 demerits', 'third-degree offense']
    if '20 demerit' in t or 'probation' in t:
        extra += ['20 demerits', 'probation']
    if 'jewelry' in t or 'dress' in t:
        extra += ['jewelry', 'wedding ring', 'dress code']
    if '10627' in t:
        extra += ['ra 10627', 'anti-bullying']
    if '11313' in t or 'safe spaces' in t:
        extra += ['ra 11313', 'safe spaces act']
    if '30 points' in t:
        extra += ['30 points', 'merit']
    base = existing or build_keywords(content, title)
    merged = base
    for e in extra:
        if e.lower() not in merged.lower():
            merged = (merged + ', ' + e).strip(', ')
    return merged


REFINE_PROMPT = """You are a precise student-handbook editor.
Return ONLY valid JSON (no markdown):

{{
  "clean_title": "short accurate section title",
  "clean_content": "cleaned text",
  "keywords": "comma-separated keywords",
  "key_facts": ["fact 1", "fact 2"]
}}

CRITICAL RULES (must follow):
1. NEVER invent any information that is not present in the Text.
2. Preserve EVERY number, demerit value, date, name, and RA number exactly.
3. If the text contains a numbered or lettered list (especially offenses, rules, goals),
   you MUST keep every single item in clean_content. Do NOT summarize or drop items.
4. If the text is mostly noise / table of contents / page numbers, set clean_content to "".
5. key_facts must be directly traceable to the Text (3–8 items max).

Title: {title}

Text:
{content}
"""


FAQ_FROM_SECTION_PROMPT = """You are converting a student handbook section into high-quality FAQ entries for a school chatbot.

RULES (must follow exactly):
1. ONLY use information that appears in the section text below. Never invent rules, numbers, names, dates, or policies.
2. Create 1–6 natural student questions that a real student would actually type (including vague / conversational phrasings).
3. Each answer must be a complete, self-contained reply that a chatbot can return directly.
4. Prefer short, clear answers. Quote key numbers, demerit values, RA numbers, or exact phrases when they matter.
5. If the section is mostly noise, table-of-contents, or pure headings with almost no substance, return an empty list.
6. Output ONLY valid JSON — no markdown, no commentary.

Output format:
{{
  "faqs": [
    {{
      "question": "natural student question (can be vague or short)",
      "answer": "clear self-contained answer drawn only from the section",
      "keywords": "comma,separated,keywords",
      "priority": 1-10
    }}
  ]
}}

Guidelines for good questions:
- Include both precise and vague versions
- Cover common student language when the section supports it
- For lists (offenses, programs, core values), make one question that asks for the whole list

Section title: {title}
Page(s): {page_start}-{page_end}

Section text:
{content}
"""


def _parse_llm_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def refine_one_chunk(idx: int, ch: Dict) -> Tuple[int, Dict]:
    fallback = dict(ch)
    title = ch.get('section_title') or 'Section'
    content = ch.get('content') or ''
    if not HAS_AI or not call_for_ingest or len(content.strip()) < 90:
        fallback['keywords'] = enrich_keywords(title, content, fallback.get('keywords') or build_keywords(content, title))
        return idx, fallback
    prompt = REFINE_PROMPT.format(title=title, content=content[:7000])
    raw = call_for_ingest(prompt, max_tokens=LLM_MAX_TOKENS)
    parsed = _parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        fallback['keywords'] = enrich_keywords(title, content, fallback.get('keywords') or build_keywords(content, title))
        return idx, fallback
    clean = (parsed.get('clean_content') or '').strip()
    if not clean:
        fallback['keywords'] = enrich_keywords(title, content, fallback.get('keywords') or build_keywords(content, title))
        return idx, fallback
    if _is_list_heavy(content) or _is_offense_list(title, content):
        orig_items = len(LIST_ITEM_RX.findall(content))
        clean_items = len(LIST_ITEM_RX.findall(clean))
        if clean_items < max(3, int(orig_items * 0.75)):
            logger.warning("LLM dropped list items for [%s] (%d → %d). Keeping original.", title[:55], orig_items, clean_items)
            clean = content
    new_title = (parsed.get('clean_title') or title)[:255]
    keywords = enrich_keywords(new_title, clean, (parsed.get('keywords') or '').strip())
    facts = parsed.get('key_facts') or []
    if facts:
        clean += '\n\nKEY FACTS:\n' + '\n'.join(f'- {f}' for f in facts)
    out = {**ch, 'section_title': new_title, 'content': clean, 'keywords': keywords}
    time.sleep(PER_CALL_PAUSE)
    return idx, out


def refine_all(chunks: List[Dict]) -> List[Dict]:
    if not chunks:
        return []
    n = len(chunks)
    logger.info("LLM refining %d chunks …", n)
    results: List[Optional[Dict]] = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(refine_one_chunk, i, ch): i for i, ch in enumerate(chunks)}
        for fut in as_completed(futures):
            try:
                idx, updated = fut.result()
                results[idx] = updated
            except Exception as e:
                i = futures[fut]
                logger.error("Refine failed on chunk %s: %s", i, e)
                results[i] = chunks[i]
            done += 1
            if done % 10 == 0 or done == n:
                logger.info("Refined %d / %d", done, n)
            if done % BATCH_COOLDOWN_EVERY == 0 and done < n:
                time.sleep(BATCH_COOLDOWN_SECONDS)
    return [r if r is not None else chunks[i] for i, r in enumerate(results)]


def section_to_faqs(ch: Dict) -> List[Dict]:
    """Convert one handbook section into FAQ-style pairs via LLM."""
    if not HAS_AI or not call_for_ingest:
        return []
    title = ch.get('section_title') or 'Section'
    content = (ch.get('content') or '').strip()
    if len(content) < 80:
        return []
    prompt = FAQ_FROM_SECTION_PROMPT.format(
        title=title,
        page_start=ch.get('page_start') or '?',
        page_end=ch.get('page_end') or '?',
        content=content[:6000],
    )
    raw = call_for_ingest(prompt, max_tokens=1200)
    parsed = _parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return []
    faqs = parsed.get('faqs') or []
    out = []
    for f in faqs:
        q = (f.get('question') or '').strip()
        a = (f.get('answer') or '').strip()
        if not q or not a or len(a) < 20:
            continue
        out.append({
            'question': q[:500],
            'answer': a[:4000],
            'keywords': (f.get('keywords') or '')[:500],
            'priority': max(1, min(10, int(f.get('priority') or 5))),
            'topic': 'Handbook',
            'source': 'handbook',
        })
    time.sleep(PER_CALL_PAUSE)
    return out


def generate_faqs_from_chunks(chunks: List[Dict], max_sections: int = 80) -> List[Dict]:
    """Run FAQ generation over the most substantial sections."""
    substantial = [c for c in chunks if len((c.get('content') or '')) >= 120]
    substantial = substantial[:max_sections]
    all_faqs: List[Dict] = []
    for i, ch in enumerate(substantial):
        logger.info("FAQ-gen %d/%d: %s", i + 1, len(substantial), (ch.get('section_title') or '')[:50])
        all_faqs.extend(section_to_faqs(ch))
    return all_faqs


def mass_insert_faqs(get_db_fn: Callable, faqs: List[Dict], retire_previous: bool = True) -> int:
    """
    Insert FAQ-style handbook pairs into the faq table.
    Sets source='handbook'. Optionally retires previous handbook FAQs first.
    """
    if not faqs:
        return 0
    conn = get_db_fn()
    cur = conn.cursor()
    inserted = 0
    try:
        if retire_previous:
            try:
                cur.execute("UPDATE faq SET is_active = 0 WHERE source = 'handbook'")
            except Exception:
                # table may not have is_active — ignore
                pass
        for f in faqs:
            q = (f.get('question') or '').strip()
            a = (f.get('answer') or '').strip()
            if not q or not a:
                continue
            kw = (f.get('keywords') or '')[:500]
            topic = (f.get('topic') or 'Handbook')[:100]
            priority = int(f.get('priority') or 5)
            try:
                cur.execute("""
                    INSERT INTO faq
                      (question, answer, topic, keywords, priority_score, source, is_active)
                    VALUES (%s, %s, %s, %s, %s, 'handbook', 1)
                """, (q[:500], a[:4000], topic, kw, priority))
                inserted += 1
            except Exception as e:
                # fallback if is_active or source columns differ
                logger.warning("mass_insert_faqs row failed: %s", e)
                try:
                    cur.execute("""
                        INSERT INTO faq (question, answer, topic, keywords, priority_score)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (q[:500], a[:4000], topic, kw, priority))
                    inserted += 1
                except Exception as e2:
                    logger.error("mass_insert_faqs fallback failed: %s", e2)
        conn.commit()
        logger.info("Mass-inserted %d handbook FAQs", inserted)
        return inserted
    finally:
        cur.close()
        conn.close()


def parse_pasted_faqs(text: str) -> List[Dict]:
    """
    Parse pasted FAQ pairs from the admin textarea.
    Accepts either:
      - JSON list of {question, answer, keywords?, priority?}
      - Simple text blocks separated by blank lines:
            Q: ...
            A: ...
    """
    text = (text or '').strip()
    if not text:
        return []
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'faqs' in data:
            data = data['faqs']
        if isinstance(data, list):
            out = []
            for f in data:
                q = (f.get('question') or f.get('q') or '').strip()
                a = (f.get('answer') or f.get('a') or '').strip()
                if q and a:
                    out.append({
                        'question': q,
                        'answer': a,
                        'keywords': f.get('keywords') or '',
                        'priority': int(f.get('priority') or 5),
                        'topic': 'Handbook',
                        'source': 'handbook',
                    })
            return out
    except Exception:
        pass

    # Simple Q:/A: blocks
    out = []
    blocks = re.split(r'\n\s*\n', text)
    for block in blocks:
        qm = re.search(r'(?:Q|Question)\s*[:\-]\s*(.+?)(?:\n|$)', block, re.I | re.S)
        am = re.search(r'(?:A|Answer)\s*[:\-]\s*(.+)', block, re.I | re.S)
        if qm and am:
            out.append({
                'question': qm.group(1).strip()[:500],
                'answer': am.group(1).strip()[:4000],
                'keywords': '',
                'priority': 5,
                'topic': 'Handbook',
                'source': 'handbook',
            })
    return out


def ingest_handbook(
    pdf_path: str,
    get_db_fn: Callable,
    use_llm_refine: bool = True,
    also_generate_faqs: bool = False,
) -> int:
    t0 = time.time()
    logger.info("Starting handbook ingestion: %s", pdf_path)
    pages = extract_pages(pdf_path)
    logger.info("Extracted %d pages", len(pages))
    chunks = split_into_sections(pages)
    logger.info("Heading split produced %d raw sections", len(chunks))
    chunks = [c for c in chunks if not is_noise_chunk(c)]
    chunks = dedupe_chunks(chunks)
    logger.info("After noise + dedupe: %d chunks", len(chunks))

    if use_llm_refine and HAS_AI:
        chunks = refine_all(chunks)
    else:
        for ch in chunks:
            ch['keywords'] = enrich_keywords(
                ch.get('section_title', ''),
                ch.get('content', ''),
                ch.get('keywords') or build_keywords(ch.get('content', ''), ch.get('section_title', ''))
            )
    chunks = [c for c in chunks if not is_noise_chunk(c)]
    chunks = dedupe_chunks(chunks)

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute('UPDATE handbook_chunks SET is_active = 0')
        for i, ch in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO handbook_chunks
                (chapter, section_title, page_start, page_end, content, keywords, chunk_order, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
                """,
                (
                    (ch.get('chapter') or '')[:200],
                    (ch.get('section_title') or '')[:255],
                    ch.get('page_start'),
                    ch.get('page_end'),
                    ch.get('content') or '',
                    ch.get('keywords') or '',
                    i,
                ),
            )
        conn.commit()
        n = len(chunks)
        logger.info("Ingested %d clean chunks in %.1fs", n, time.time() - t0)

        if also_generate_faqs and HAS_AI:
            faqs = generate_faqs_from_chunks(chunks)
            mass_insert_faqs(get_db_fn, faqs, retire_previous=True)

        return n
    finally:
        cur.close()
        conn.close()


def import_chunks_json(get_db_fn: Callable, data: dict) -> int:
    """Import a pre-chunked handbook_chunks_*.json export."""
    chunks = data.get('chunks') or []
    if not chunks:
        return 0
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute('UPDATE handbook_chunks SET is_active = 0')
        n = 0
        for i, ch in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO handbook_chunks
                (chapter, section_title, page_start, page_end, content, keywords, chunk_order, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
                """,
                (
                    (ch.get('chapter') or '')[:200],
                    (ch.get('section_title') or '')[:255],
                    ch.get('page_start'),
                    ch.get('page_end'),
                    ch.get('content') or '',
                    ch.get('keywords') or '',
                    i,
                ),
            )
            n += 1
        conn.commit()
        return n
    finally:
        cur.close()
        conn.close()