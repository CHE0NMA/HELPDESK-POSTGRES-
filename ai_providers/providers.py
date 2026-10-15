"""All outbound AI-provider calls (Groq + Gemini).
Supports multiple Groq API keys and multiple Gemini API keys with:
- least-used rotation
- daily call limits (Groq)
- automatic cooldown on 429
"""

import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Optional

from groq.types.chat import ChatCompletionUserMessageParam

from config import (
    AI_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
)

INGEST_GROQ_MODEL = os.getenv("INGEST_GROQ_MODEL", "openai/gpt-oss-20b")
FAST_GROQ_MODEL = os.getenv("FAST_GROQ_MODEL", INGEST_GROQ_MODEL)

from .circuit_breaker import circuit_breaker
from .metrics import model_metrics

logger = logging.getLogger("pac_helpdesk.ai_providers")

# ---------------------------------------------------------------------
# Groq keys
# ---------------------------------------------------------------------
GROQ_API_KEYS = [
    GROQ_API_KEY,
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
    os.getenv("GROQ_API_KEY_6", ""),
    os.getenv("GROQ_API_KEY_7", ""),
    os.getenv("GROQ_API_KEY_8", ""),
]
GROQ_API_KEYS = [k.strip() for k in GROQ_API_KEYS if k and k.strip()]

# ---------------------------------------------------------------------
# Gemini keys
# ---------------------------------------------------------------------
GEMINI_API_KEYS = [
    GEMINI_API_KEY,
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
    os.getenv("GEMINI_API_KEY_5", ""),
    os.getenv("GEMINI_API_KEY_6", ""),
    os.getenv("GEMINI_API_KEY_7", ""),
    os.getenv("GEMINI_API_KEY_8", ""),
]
GEMINI_API_KEYS = [k.strip() for k in GEMINI_API_KEYS if k and k.strip()]


SOFT_DAILY_LIMIT = 400
HARD_DAILY_LIMIT = 1000
COOLDOWN_MINUTES = 35

GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "12"))
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))


# ---------------------------------------------------------------------
# Groq key management
# ---------------------------------------------------------------------
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


def _mark_key_cooldown(key: str, reason: str, minutes: int = COOLDOWN_MINUTES):
    with _key_lock:
        _key_cooldown[key] = datetime.now() + timedelta(minutes=minutes)
        logger.warning(
            f"Groq key ...{key[-6:]} {reason} → cooldown {minutes} min. "
            f"Today's usage: {_key_usage_today[key]}"
        )


def _mark_key_rate_limited(key: str, minutes: int = COOLDOWN_MINUTES):
    _mark_key_cooldown(key, "rate-limited", minutes)


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


# ---------------------------------------------------------------------
# Gemini key management
# ---------------------------------------------------------------------
_gemini_key_lock = threading.Lock()
_gemini_usage_today = defaultdict(int)
_gemini_last_reset = date.today()
_gemini_cooldown = {}


def _reset_gemini_daily_if_needed():
    global _gemini_last_reset
    today = date.today()
    if today != _gemini_last_reset:
        _gemini_usage_today.clear()
        _gemini_last_reset = today
        logger.info("Gemini daily usage counters reset.")


def get_next_gemini_key():
    """Return the least-used Gemini key that is not on cooldown."""
    if not GEMINI_API_KEYS:
        return None

    now = datetime.now()
    with _gemini_key_lock:
        _reset_gemini_daily_if_needed()

        usable = []
        for k in GEMINI_API_KEYS:
            cd = _gemini_cooldown.get(k)
            if cd and now < cd:
                continue
            usable.append(k)

        if not usable:
            return None

        # least-used
        usable.sort(key=lambda k: _gemini_usage_today[k])
        return usable[0]


def mark_gemini_success(key: str):
    with _gemini_key_lock:
        _gemini_usage_today[key] += 1


def _mark_gemini_cooldown(key: str, reason: str, minutes: int = 30):
    with _gemini_key_lock:
        _gemini_cooldown[key] = datetime.now() + timedelta(minutes=minutes)
        logger.warning(
            f"Gemini key ...{key[-6:]} {reason} → cooldown {minutes} min. "
            f"Today's usage: {_gemini_usage_today[key]}"
        )


def mark_gemini_rate_limited(key: str, minutes: int = 30):
    _mark_gemini_cooldown(key, "rate-limited", minutes)


def mark_gemini_auth_failed(key: str, minutes: int = 30):
    _mark_gemini_cooldown(key, "authentication failed", minutes)


def get_gemini_usage_report() -> dict:
    """Useful for debugging / admin dashboard."""
    with _gemini_key_lock:
        _reset_gemini_daily_if_needed()
        return {
            "date": str(_gemini_last_reset),
            "keys": {
                f"...{k[-6:]}": {
                    "calls_today": _gemini_usage_today[k],
                    "on_cooldown": bool(
                        _gemini_cooldown.get(k) and datetime.now() < _gemini_cooldown[k]
                    ),
                }
                for k in GEMINI_API_KEYS
            },
            "total_calls_today": sum(_gemini_usage_today.values()),
        }


current_model = f"{AI_PROVIDER}:{GROQ_MODEL if AI_PROVIDER == 'groq' else GEMINI_MODEL}"
FALLBACK_ORDER = ['groq', 'gemini']


# ---------------------------------------------------------------------
# Actual provider calls
# ---------------------------------------------------------------------
def _call_groq(prompt: str, max_tokens: int = 400,
               model: Optional[str] = None) -> str:
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
            client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS)
            messages: list[ChatCompletionUserMessageParam] = [
                {"role": "user", "content": prompt}
            ]
            if use_model.startswith("openai/gpt-oss-"):
                resp = client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    max_completion_tokens=max(max_tokens, 64),
                    temperature=0.1,
                    reasoning_effort="low",
                    include_reasoning=False,
                )
            else:
                resp = client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
            _mark_key_success(api_key)
            content = resp.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "401" in err_str or "403" in err_str or "invalid_api_key" in err_str:
                _mark_key_cooldown(api_key, "authentication failed")
                continue
            if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                _mark_key_rate_limited(api_key)
                continue
            if "timeout" in err_str or "timed out" in err_str:
                logger.warning(f"Groq timeout (>{GROQ_TIMEOUT_SECONDS}s) on ...{api_key[-6:]}: {e}")
                return ""
            logger.error(f"Groq error on ...{api_key[-6:]}: {e}")
            return ""

    if last_error:
        logger.error(f"All Groq keys exhausted or failed. Last error: {last_error}")
        logger.info(f"Usage report: {get_groq_usage_report()}")
    return ""


def _call_gemini(prompt: str, max_tokens: int = 400) -> str:
    if not GEMINI_API_KEYS:
        logger.error("No Gemini API keys configured")
        return ""

    last_error = None

    for _ in range(len(GEMINI_API_KEYS)):
        api_key = get_next_gemini_key()
        if not api_key:
            break

        try:
            from google import genai
            from google.genai import types

            with genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(GEMINI_TIMEOUT_SECONDS * 1000)
                ),
            ) as client:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                        automatic_function_calling=(
                            types.AutomaticFunctionCallingConfig(disable=True)
                        ),
                    ),
                )
            text = (resp.text or "").strip()
            if not text:
                logger.warning("Gemini returned an empty response")
                return ""
            mark_gemini_success(api_key)
            return text

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if ("401" in err_str or "403" in err_str or "invalid_api_key" in err_str
                    or "invalid api key" in err_str or "api key not valid" in err_str):
                mark_gemini_auth_failed(api_key)
                continue
            if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                mark_gemini_rate_limited(api_key)
                continue
            if "timeout" in err_str or "timed out" in err_str:
                logger.warning(f"Gemini timeout (>{GEMINI_TIMEOUT_SECONDS}s) on ...{api_key[-6:]}: {e}")
                return ""
            logger.error(f"Gemini error on ...{api_key[-6:]}: {e}")
            return ""

    if last_error:
        logger.error(f"All Gemini keys exhausted or failed. Last error: {last_error}")
        logger.info(f"Gemini usage report: {get_gemini_usage_report()}")
    return ""


def _call_groq_fast(prompt: str, max_tokens: int = 400) -> str:
    """Same as _call_groq but pinned to FAST_GROQ_MODEL."""
    return _call_groq(prompt, max_tokens=max_tokens, model=FAST_GROQ_MODEL)


_PROVIDER_FUNCS = {
    'groq': _call_groq,
    'gemini': _call_gemini,
}

_FAST_PROVIDER_FUNCS = {
    'groq': _call_groq_fast,
    'gemini': _call_gemini,
}

_PROVIDER_READY = {
    'groq': lambda: bool(GROQ_API_KEYS),
    'gemini': lambda: bool(GEMINI_API_KEYS),
}


def _run_provider_chain(prompt: str, max_tokens: int, order, provider_funcs, label: str) -> str:
    """Shared body for call_with_fallback / call_gemini_then_groq /
    call_fast_with_fallback.
    """
    tried = []
    for provider in order:
        if not _PROVIDER_READY.get(provider, lambda: False)():
            continue
        tried.append(provider)
        fn = provider_funcs.get(provider)
        if not fn:
            continue

        t0 = time.perf_counter()
        result = fn(prompt, max_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result:
            model_metrics.record_response_time(elapsed_ms, True, provider=provider)
            circuit_breaker.record_success()
            return result
        model_metrics.record_response_time(elapsed_ms, False, provider=provider)
        circuit_breaker.record_failure()
        logger.warning(f"Provider '{provider}' failed during {label} (tried: {tried})")

    logger.error(f"{label}: all providers failed. Tried: {tried}")
    return ""


def call_for_ingest(prompt: str, max_tokens: int = 2200) -> str:
    """
    Bulk ingest helper:
    1) Groq with llama-3.1-8b-instant (higher free limits)
    2) Gemini fallback
    """
    # 1) Groq 8B
    if GROQ_API_KEYS:
        t0 = time.perf_counter()
        result = _call_groq(prompt, max_tokens=max_tokens, model=INGEST_GROQ_MODEL)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if result:
            model_metrics.record_response_time(elapsed_ms, True, provider="groq")
            circuit_breaker.record_success()
            return result
        model_metrics.record_response_time(elapsed_ms, False, provider="groq")
        circuit_breaker.record_failure()
        logger.warning("Ingest: Groq 8B failed, trying Gemini")

    # 2) Gemini
    if GEMINI_API_KEYS:
        t0 = time.perf_counter()
        result = _call_gemini(prompt, max_tokens=max_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if result:
            model_metrics.record_response_time(elapsed_ms, True, provider="gemini")
            circuit_breaker.record_success()
            return result
        model_metrics.record_response_time(elapsed_ms, False, provider="gemini")
        circuit_breaker.record_failure()
        logger.warning("Ingest: Gemini also failed")

    logger.error("Ingest: all providers failed (8B + Gemini)")
    return ""


def call_with_fallback(prompt: str, max_tokens: int = 400) -> str:
    order = [AI_PROVIDER] + [p for p in FALLBACK_ORDER if p != AI_PROVIDER]
    return _run_provider_chain(prompt, max_tokens, order, _PROVIDER_FUNCS, "call_with_fallback")


def call_gemini_then_groq(prompt: str, max_tokens: int = 400) -> str:
    """Prefer Gemini, then fall back to Groq."""
    return _run_provider_chain(
        prompt, max_tokens, ("gemini", "groq"), _PROVIDER_FUNCS, "gemini-then-groq"
    )


def call_fast_with_fallback(prompt: str, max_tokens: int = 60) -> str:
    """Cheap/fast-model variant of call_with_fallback."""
    order = [AI_PROVIDER] + [p for p in FALLBACK_ORDER if p != AI_PROVIDER]
    return _run_provider_chain(
        prompt, max_tokens, order, _FAST_PROVIDER_FUNCS, "call_fast_with_fallback"
    )


def llm_chat(prompt: str, max_tokens: int = 400) -> str:
    """Main entry point used by the rest of the app."""
    if circuit_breaker.is_open():
        return ""
    return call_with_fallback(prompt, max_tokens)


def llm_chat_fast(prompt: str, max_tokens: int = 60) -> str:
    """Entry point for short, low-stakes classification calls."""
    if circuit_breaker.is_open():
        return ""
    return call_fast_with_fallback(prompt, max_tokens)