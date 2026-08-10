"""Per-session conversation memory (short-term context for follow-ups)."""
import re
import time
from typing import Dict, List, Optional


class ConversationContext:
    def __init__(self):
        self.last_topic = None
        self.last_intent = None
        self.conversation_history: List[dict] = []
        self.last_faq_answer = None
        self.last_faq_question = None
        self.last_answer = None
        self.last_schedule_query = None
        self.last_detected_lang = None

        self.last_disambiguation: Optional[dict] = None

    def update(self, topic=None, intent=None, user_msg=None, bot_response=None,
               faq_answer=None, faq_question=None, detected_lang=None):
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

        if user_msg and bot_response:
            self.conversation_history.append({
                'user': user_msg, 'bot': bot_response, 'ts': time.time(),
            })
            # Keep a bit more than the 3 exchanges we promise to "remember"
            # so trimming never eats into the guaranteed window.
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)
            self.last_answer = bot_response

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

    def clear_disambiguation(self):
        self.last_disambiguation = None


conversation_contexts: Dict[str, ConversationContext] = {}


def get_conversation_context(session_id: str) -> ConversationContext:
    if session_id not in conversation_contexts:
        conversation_contexts[session_id] = ConversationContext()
    return conversation_contexts[session_id]


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