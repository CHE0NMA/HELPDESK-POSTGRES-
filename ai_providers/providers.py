"""All outbound AI-provider calls (Groq + Gemini).
Supports multiple Groq API keys with:
- least-used rotation
- daily call limits
- automatic cooldown on 429
"""

import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta, date

from config import (
    AI_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
)
# BUG FIX: llama-3.1-8b-instant shuts down 08/16/26 (Groq's own
# deprecation page: https://console.groq.com/docs/deprecations) --
# Groq's recommended replacement is openai/gpt-oss-20b. Same account,
# same GROQ_API_KEY(s) -- no new API registration needed, just the model
# string. If INGEST_GROQ_MODEL is set as an env var in your deployment,
# update it there too; this default only applies when it's unset.
INGEST_GROQ_MODEL = os.getenv("INGEST_GROQ_MODEL", "openai/gpt-oss-20b")
from .circuit_breaker import circuit_breaker
from .metrics import model_metrics

logger = logging.getLogger("pac_helpdesk.ai_providers")

GROQ_API_KEYS = [
    GROQ_API_KEY,
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
    os.getenv("GROQ_API_KEY_6", ""),
    os.getenv("GROQ_API_KEY_7", ""),
    os.getenv("GROQ_API_KEY_8", ""),
    os.getenv("GROQ_API_KEY_9", ""),
    os.getenv("GROQ_API_KEY_10", ""),
    os.getenv("GROQ_API_KEY_11", ""),
    os.getenv("GROQ_API_KEY_12", ""),
    os.getenv("GROQ_API_KEY_13", ""),
    os.getenv("GROQ_API_KEY_14", ""),
    os.getenv("GROQ_API_KEY_15", ""),
    os.getenv("GROQ_API_KEY_16", ""),
    os.getenv("GROQ_API_KEY_17", ""),
    os.getenv("GROQ_API_KEY_18", ""),
    os.getenv("GROQ_API_KEY_19", ""),
    os.getenv("GROQ_API_KEY_20", ""),
]
GROQ_API_KEYS = [k.strip() for k in GROQ_API_KEYS if k and k.strip()]


SOFT_DAILY_LIMIT = 150
HARD_DAILY_LIMIT = 200
COOLDOWN_MINUTES = 35


_key_lock = threading.Lock()
_key_usage_today = defaultdict(int)
_key_last_reset = date.today()
_key_cooldown = {}


def _reset_daily_counters_if_needed():
    global _key_last_reset
    today = date.today()
    if today != _key_last_reset:
        _key_usage_today.clear()
        _key_last_reset = today
        logger.info("Groq daily usage counters reset.")


def _get_next_groq_key():
    """Return the least-used key that is still under limits and not on cooldown."""
    if not GROQ_API_KEYS:
        return None

    now = datetime.now()
    with _key_lock:
        _reset_daily_counters_if_needed()

        usable = []
        for k in GROQ_API_KEYS:

            cd = _key_cooldown.get(k)
            if cd and now < cd:
                continue

            if _key_usage_today[k] >= HARD_DAILY_LIMIT:
                continue
            usable.append(k)

        if not usable:
            return None

        under_soft = [k for k in usable if _key_usage_today[k] < SOFT_DAILY_LIMIT]
        candidates = under_soft if under_soft else usable
        candidates.sort(key=lambda k: _key_usage_today[k])
        return candidates[0]


def _mark_key_success(key: str):
    with _key_lock:
        _key_usage_today[key] += 1


def _mark_key_rate_limited(key: str, minutes: int = COOLDOWN_MINUTES):
    with _key_lock:
        _key_cooldown[key] = datetime.now() + timedelta(minutes=minutes)
        logger.warning(
            f"Groq key ...{key[-6:]} rate-limited → cooldown {minutes} min. "
            f"Today's usage: {_key_usage_today[key]}"
        )


def get_groq_usage_report() -> dict:
    """Useful for debugging / admin dashboard."""
    with _key_lock:
        _reset_daily_counters_if_needed()
        return {
            "date": str(_key_last_reset),
            "keys": {
                f"...{k[-6:]}": {
                    "calls_today": _key_usage_today[k],
                    "on_cooldown": bool(
                        _key_cooldown.get(k) and datetime.now() < _key_cooldown[k]
                    ),
                }
                for k in GROQ_API_KEYS
            },
            "total_calls_today": sum(_key_usage_today.values()),
        }


current_model = f"{AI_PROVIDER}:{GROQ_MODEL if AI_PROVIDER == 'groq' else GEMINI_MODEL}"
FALLBACK_ORDER = ['groq', 'gemini']

_gemini_model = None


def _get_gemini():
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model


def _call_groq(prompt: str, max_tokens: int = 400, model: str = None) -> str:
    if not GROQ_API_KEYS:
        logger.error("No Groq API keys configured")
        return ""

    use_model = model or GROQ_MODEL
    last_error = None

    for _ in range(len(GROQ_API_KEYS)):
        api_key = _get_next_groq_key()
        if not api_key:
            break

        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            _mark_key_success(api_key)
            return resp.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                _mark_key_rate_limited(api_key)
                continue
            logger.error(f"Groq error on ...{api_key[-6:]}: {e}")
            return ""

    if last_error:
        logger.error(f"All Groq keys exhausted or failed. Last error: {last_error}")
        logger.info(f"Usage report: {get_groq_usage_report()}")
    return ""

def _call_gemini(prompt: str, max_tokens: int = 400) -> str:
    try:
        import google.generativeai as genai
        resp = _get_gemini().generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
            ),
        )
        return resp.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return ""


_PROVIDER_FUNCS = {
    'groq': _call_groq,
    'gemini': _call_gemini,
}

_PROVIDER_READY = {
    'groq': lambda: bool(GROQ_API_KEYS),
    'gemini': lambda: bool(GEMINI_API_KEY),
}

def call_for_ingest(prompt: str, max_tokens: int = 2200) -> str:
    """
    Bulk ingest helper:
    1) Groq with llama-3.1-8b-instant (higher free limits)
    2) Gemini fallback
    Does NOT use 70B.
    """
    # 1) Groq 8B
    if GROQ_API_KEYS:
        result = _call_groq(prompt, max_tokens=max_tokens, model=INGEST_GROQ_MODEL)
        if result:
            model_metrics.record_response_time(0, True)
            circuit_breaker.record_success()
            return result
        model_metrics.record_response_time(0, False)
        circuit_breaker.record_failure()
        logger.warning("Ingest: Groq 8B failed, trying Gemini")

    # 2) Gemini
    if GEMINI_API_KEY:
        result = _call_gemini(prompt, max_tokens=max_tokens)
        if result:
            model_metrics.record_response_time(0, True)
            circuit_breaker.record_success()
            return result
        model_metrics.record_response_time(0, False)
        circuit_breaker.record_failure()
        logger.warning("Ingest: Gemini also failed")

    logger.error("Ingest: all providers failed (8B + Gemini)")
    return ""


def call_with_fallback(prompt: str, max_tokens: int = 400) -> str:
    tried = []
    order = [AI_PROVIDER] + [p for p in FALLBACK_ORDER if p != AI_PROVIDER]

    for provider in order:
        if not _PROVIDER_READY.get(provider, lambda: False)():
            continue
        tried.append(provider)
        fn = _PROVIDER_FUNCS.get(provider)
        if not fn:
            continue
        result = fn(prompt, max_tokens)
        if result:
            model_metrics.record_response_time(0, True)
            circuit_breaker.record_success()
            return result
        model_metrics.record_response_time(0, False)
        circuit_breaker.record_failure()
        logger.warning(f"Provider '{provider}' failed (tried: {tried})")

    logger.error(f"All AI providers failed. Tried: {tried}")
    return ""

def call_gemini_then_groq(prompt: str, max_tokens: int = 400) -> str:
    """
    Prefer Gemini, then fall back to Groq.
    Useful for bulk jobs (handbook ingest) so Groq daily limits last longer.
    """
    tried = []
    for provider in ("gemini", "groq"):
        if not _PROVIDER_READY.get(provider, lambda: False)():
            continue
        tried.append(provider)
        fn = _PROVIDER_FUNCS.get(provider)
        if not fn:
            continue
        result = fn(prompt, max_tokens)
        if result:
            model_metrics.record_response_time(0, True)
            circuit_breaker.record_success()
            return result
        model_metrics.record_response_time(0, False)
        circuit_breaker.record_failure()
        logger.warning(f"Provider '{provider}' failed during gemini-then-groq (tried: {tried})")

    logger.error(f"Gemini+Groq both failed. Tried: {tried}")
    return ""

def llm_chat(prompt: str, max_tokens: int = 400) -> str:
    """Main entry point used by the rest of the app (faq, handbook, dates…)."""
    if circuit_breaker.is_open():
        return ""
    return call_with_fallback(prompt, max_tokens)