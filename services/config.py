"""
services/config.py — the ONE place institution-specific naming lives.

Previously "PAC Bot" and "Philippine Advent College (PAC)" were hardcoded
directly into LLM prompts inside answer_engine.py. That meant swapping in a
different school's handbook still had the old school's name baked into
every AI answer. Everything now reads from here, and this reads from env
vars first, falling back to values auto-detected from the handbook itself
during ingestion (see `detect_institution_name` below), and only then to
generic defaults.
"""
import os
import re
from typing import Optional

BOT_NAME = os.environ.get("HELPDESK_BOT_NAME", "Campus Bot")
INSTITUTION_NAME = os.environ.get("HELPDESK_INSTITUTION_NAME", "").strip()
INSTITUTION_ABBREVIATION = os.environ.get("HELPDESK_INSTITUTION_ABBR", "").strip()


def detect_institution_name(sample_text: str) -> Optional[str]:
    """Best-effort extraction of "Full Institution Name (ABBR)" from the
    handbook's own front matter, used at ingestion time to auto-populate
    config when HELPDESK_INSTITUTION_NAME isn't set. Never invents a name —
    returns None if nothing confident is found, in which case callers
    should fall back to a generic phrase ("the school", "this institution").
    """
    if not sample_text:
        return None
    m = re.search(
        r'([A-Z][A-Za-z.,\'\-\s]{5,80}?(College|University|School|Academy|Institute))\s*(?:\(([A-Z]{2,6})\))?',
        sample_text,
    )
    if not m:
        return None
    return m.group(1).strip()


def institution_label() -> str:
    """What to call the school in prompts/UI. Falls back to a neutral
    phrase rather than a hardcoded name when nothing has been configured
    or detected — this is what makes the rest of the codebase portable."""
    if INSTITUTION_NAME:
        if INSTITUTION_ABBREVIATION:
            return f"{INSTITUTION_NAME} ({INSTITUTION_ABBREVIATION})"
        return INSTITUTION_NAME
    return "the school"


def bot_identity_line() -> str:
    """One-line self-identification used at the top of LLM prompts."""
    return f"You are {BOT_NAME}, the official student helpdesk assistant for {institution_label()}."