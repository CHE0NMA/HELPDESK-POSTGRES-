"""Per-session conversation memory (short-term context for follow-ups).

--- "random new chat every time" fix ---
This used to be backed ONLY by the module-level `conversation_contexts`
dict below — a plain Python dict living in ONE worker process's memory.
That silently breaks in two extremely common deployment situations,
both of which look exactly like "randomly starts a new chat":

  1. More than one worker process (gunicorn/uwsgi `--workers > 1`, or
     any autoscaled/multi-dyno host). Each worker has its OWN copy of
     this dict. Request 1 (page load) can land on worker A and create
     the session there; request 2 (the student's very next message)
     gets load-balanced to worker B, which has never heard of that
     session_id — so it looks like a brand new chat, and which worker
     you land on next is effectively random from the student's POV.
  2. Any restart/redeploy/idle-sleep of the process (common on
     free/hobby hosts) wipes the dict entirely — every open
     conversation resets at once.

Neither of those is fixable by changing anything client-side (cookies,
localStorage session ids, etc.) — the state has to actually be shared
and durable on the server. Fixed here by making Postgres the source of
truth: every mutation is written through to a `conversation_state`
table (session_id, JSON payload), and a cache-miss on the in-process
dict falls back to loading from the DB before creating a fresh context.
The in-process dict is kept as a fast L1 cache so a hot session doesn't
round-trip to Postgres on every single message.

If the DB write/read fails for any reason, everything degrades to the
old in-memory-only behavior rather than raising — a slow/unavailable DB
should never break the chat itself, it just loses the durability fix.

NOTE: `last_disambiguation` is intentionally NOT persisted — its
payload can carry raw handbook-chunk dicts / FAQ row tuples that aren't
cleanly JSON round-trippable, and a "which one did you mean?" prompt is
answered (or abandoned) within the same short exchange in practice, so
losing it on a cross-worker miss is a much smaller problem than losing
the whole conversation.
"""
import json
import logging
import re
import time
from datetime import date, datetime
from typing import Dict, List, Optional

logger = logging.getLogger("pac_helpdesk.conversation")

_TABLE_READY = False


_L1_TTL_SECONDS = 8  


class ConversationContext:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.last_topic = None
        self.last_intent = None
        self.conversation_history: List[dict] = []
        self.last_faq_answer = None
        self.last_faq_question = None
        self.last_answer = None
        self.last_schedule_query = None
        self.last_detected_lang = None
        self.last_handbook_sections: List[str] = []

        self.last_disambiguation: Optional[dict] = None
        self._loaded_at = time.time()

    def update(self, topic=None, intent=None, user_msg=None, bot_response=None,
               faq_answer=None, faq_question=None, detected_lang=None,
               handbook_sections=None, clear_schedule_context=False):
        if topic:
            self.last_topic = topic
        if intent:
            self.last_intent = intent
        if faq_answer:
            self.last_faq_answer = faq_answer
        if faq_question:
            self.last_faq_question = faq_question
        if detected_lang:
            self.last_detected_lang = detected_lang
        if handbook_sections:
            self.last_handbook_sections = list(handbook_sections)[:3]

        if clear_schedule_context:
            self.last_schedule_query = None

        if user_msg and bot_response:
            self.conversation_history.append({
                'user': user_msg, 'bot': bot_response, 'ts': time.time(),
            })
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)
            self.last_answer = bot_response

        _save_ctx(self.session_id, self)

    def get_history(self, limit: int = 3) -> List[dict]:
        return self.conversation_history[-limit:]

    def get_context_summary(self) -> str:
        if not self.conversation_history:
            return ""
        lines = ["Previous conversation:"]
        for ex in self.conversation_history[-3:]:
            lines.append(f"Student: {ex['user']}")
            lines.append(f"Assistant: {ex['bot'][:200]}")
        if self.last_faq_question:
            lines.append(f"\nLast topic discussed: {self.last_faq_question}")
        if self.last_topic:
            lines.append(f"Topic category: {self.last_topic}")
        if self.last_handbook_sections:
            lines.append(
                "Last handbook section(s) discussed: "
                + ", ".join(self.last_handbook_sections)
            )
        return "\n".join(lines)

    def is_followup(self, question: str) -> bool:
        if not self.conversation_history:
            return False
        q = question.lower().strip()
        words = q.split()
        if not words:
            return False
        if len(words) <= 2:
            return True
        followup_phrases = [
            'and the', 'and if', 'also', 'how about', 'what about', 'then what',
            'after that', 'what if', 'so that means', 'so do i', 'so can i',
            'so will i', 'do i also', 'is it required', 'do i need to',
            'ano naman', 'paano naman', 'tapos', 'so ibig sabihin',
            'so meaning', 'meaning', 'ibig sabihin', 'so', 'pano naman',
            'ug ang', 'unya', 'then', 'ok so', 'about that', 'regarding',
            'paano', 'unsaon'
        ]
        for phrase in followup_phrases:
            if q.startswith(phrase) or f' {phrase} ' in q:
                return True
        return False

    def set_disambiguation(self, dtype: str, options: List[dict], original_question: str,
                            detected_lang: str = 'english'):
        """Remember a "which one did you mean?" prompt.

        dtype: 'handbook' | 'faq'
        options: list of dicts, each MUST have a 'label' key (the text
            shown to the student) plus whatever payload the resolver for
            that type needs (e.g. the handbook chunk dict, or the faq row).
        original_question: the question that *triggered* the disambiguation
            — reused when answering the selected option so the answer stays
            grounded in what the student actually wanted to know.
        """
        self.last_disambiguation = {
            'type': dtype,
            'options': options,
            'original_question': original_question,
            'detected_lang': detected_lang,
        }

        _save_ctx(self.session_id, self)

    def clear_disambiguation(self):
        self.last_disambiguation = None
        _save_ctx(self.session_id, self)


conversation_contexts: Dict[str, ConversationContext] = {}


def get_conversation_context(session_id: str) -> ConversationContext:
    session_id = session_id or 'default'
    ctx = conversation_contexts.get(session_id)
    if ctx is not None and (time.time() - ctx._loaded_at) < _L1_TTL_SECONDS:
        return ctx

    ctx = _load_ctx(session_id)
    conversation_contexts[session_id] = ctx
    return ctx

def _ensure_table():
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_state (
                    session_id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            conn.commit()
            _TABLE_READY = True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        
        logger.debug(f"conversation_state ensure_table skipped: {e}")


def _json_default(obj):
    """Handles the one non-JSON-native type we actually store: dates."""
    if isinstance(obj, (date, datetime)):
        return {'__date__': obj.isoformat()}
    return str(obj)


def _json_object_hook(d: dict):
    if set(d.keys()) == {'__date__'}:
        try:
            return date.fromisoformat(d['__date__'])
        except ValueError:
            return None
    return d


def _ctx_to_dict(ctx: 'ConversationContext') -> dict:
    return {
        'last_topic': ctx.last_topic,
        'last_intent': ctx.last_intent,
        'conversation_history': ctx.conversation_history,
        'last_faq_answer': ctx.last_faq_answer,
        'last_faq_question': ctx.last_faq_question,
        'last_answer': ctx.last_answer,
        'last_schedule_query': ctx.last_schedule_query,
        'last_detected_lang': ctx.last_detected_lang,
        'last_handbook_sections': ctx.last_handbook_sections,
        # last_disambiguation deliberately omitted — see module docstring.
    }


def _apply_dict_to_ctx(ctx: 'ConversationContext', data: dict) -> None:
    ctx.last_topic = data.get('last_topic')
    ctx.last_intent = data.get('last_intent')
    ctx.conversation_history = data.get('conversation_history') or []
    ctx.last_faq_answer = data.get('last_faq_answer')
    ctx.last_faq_question = data.get('last_faq_question')
    ctx.last_answer = data.get('last_answer')
    ctx.last_schedule_query = data.get('last_schedule_query')
    ctx.last_detected_lang = data.get('last_detected_lang')
    ctx.last_handbook_sections = data.get('last_handbook_sections') or []


def _save_ctx(session_id: Optional[str], ctx: 'ConversationContext') -> None:
    if not session_id:
        return
    _ensure_table()
    if not _TABLE_READY:
        return
    try:
        from database import get_db
        payload = json.dumps(_ctx_to_dict(ctx), default=_json_default)
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO conversation_state (session_id, data, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (session_id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = NOW()
            """, (session_id, payload))
            conn.commit()
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"_save_ctx failed for session={session_id}: {e}")


def delete_ctx(session_id: str) -> None:
    """Fully wipe a session's conversation memory — both the in-process
    L1 cache and the durable DB row. Use this (not just popping the
    in-memory dict) whenever a chat is explicitly reset/abandoned
    (routes/pages.py's new_chat() / _reset_session_for_new_chat()), so a
    reused or recycled session_id can never resurrect a stale
    conversation from Postgres after the visible chat was supposed to
    have been cleared.
    """
    if not session_id:
        return
    conversation_contexts.pop(session_id, None)
    _ensure_table()
    if not _TABLE_READY:
        return
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM conversation_state WHERE session_id = %s", (session_id,))
            conn.commit()
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"delete_ctx failed for session={session_id}: {e}")


def _load_ctx(session_id: str) -> 'ConversationContext':
    ctx = ConversationContext(session_id=session_id)
    _ensure_table()
    if not _TABLE_READY:
        return ctx
    try:
        from database import get_db
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT data FROM conversation_state WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
            conn.close()
        if row and row[0] is not None:
            raw = row[0]

            data = (
                json.loads(raw, object_hook=_json_object_hook)
                if isinstance(raw, str) else _rehydrate_dates(raw)
            )
            _apply_dict_to_ctx(ctx, data)
    except Exception as e:
        logger.debug(f"_load_ctx failed for session={session_id}: {e}")
    return ctx


def _rehydrate_dates(obj):
    """Walk an already-parsed JSONB dict/list and convert any
    {'__date__': 'YYYY-MM-DD'} markers back into real date objects —
    needed when psycopg2 hands back JSONB pre-parsed (so our
    json.loads object_hook never ran)."""
    if isinstance(obj, dict):
        if set(obj.keys()) == {'__date__'}:
            try:
                return date.fromisoformat(obj['__date__'])
            except (ValueError, TypeError):
                return None
        return {k: _rehydrate_dates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rehydrate_dates(v) for v in obj]
    return obj


def build_context_snapshot(ctx) -> str:
    if not ctx or not getattr(ctx, 'conversation_history', None):
        return ''
    lines = []
    for ex in ctx.get_history(3):
        lines.append(f"Student: {ex.get('user', '')}")
        bot = (ex.get('bot', '') or '')
        lines.append(f"Assistant: {bot[:300]}")
    return "\n".join(lines)


def resolve_disambiguation_choice(user_reply: str, disambiguation: Optional[dict],
                                   threshold: int = 70) -> Optional[dict]:
    """Given a pending disambiguation prompt and the student's next
    message, figure out which option (if any) they picked.

    Matches on:
      - a leading number ("2", "2.", "#2 please")
      - fuzzy overlap between the reply and an option's label (handles the
        student pasting the option text back, with or without the page
        range, punctuation, etc.)

    Returns the matching option dict, or None if the reply doesn't look
    like a selection at all — the caller should then treat the message as
    a brand new question and clear the pending state.
    """
    if not disambiguation:
        return None
    options = disambiguation.get('options') or []
    if not options:
        return None

    reply = user_reply.strip()
    if not reply:
        return None

    m = re.match(r'^\s*#?\s*(\d+)\b', reply)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        return None

    reply_norm = reply.lower()
    best_opt, best_score = None, 0
    for opt in options:
        label = (opt.get('label') or '').lower()
        score = max(
            fuzz.token_set_ratio(reply_norm, label),
            fuzz.partial_ratio(reply_norm, label),
        )
        if score > best_score:
            best_opt, best_score = opt, score

    if best_score >= threshold:
        return best_opt
    return None