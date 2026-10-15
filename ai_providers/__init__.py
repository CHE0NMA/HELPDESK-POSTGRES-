"""ai_providers package — AI provider calls (Groq/Gemini), caching,
circuit breaker, and response-time metrics, shared by every service
(faq, handbook, dates, corrections, ai_retrieval, self_heal, ...) and by
routes/schedules.py's cache invalidation.

This is what makes `from ai_providers import ai_cache, circuit_breaker,
llm_chat` (etc.) work anywhere else in the app — without this file,
`ai_providers/` is just a folder, not an importable package, and
`cache.py`'s own `from .metrics import model_metrics` has nothing to
resolve against either.
"""
from .cache import AICache, FAQCache, ai_cache, faq_cache
from .circuit_breaker import AICircuitBreaker, circuit_breaker
from .metrics import ModelMetrics, model_metrics
from .providers import (
    llm_chat,
    llm_chat_fast,
    call_with_fallback,
    call_fast_with_fallback,
    call_gemini_then_groq,
    call_for_ingest,
    get_groq_usage_report,
    get_gemini_usage_report,
    get_next_gemini_key,
    mark_gemini_success,
    mark_gemini_rate_limited,
    mark_gemini_auth_failed,
    current_model,
    GROQ_API_KEYS,
    GEMINI_API_KEYS,
)

__all__ = [
    "AICache", "FAQCache", "ai_cache", "faq_cache",
    "AICircuitBreaker", "circuit_breaker",
    "ModelMetrics", "model_metrics",
    "llm_chat", "llm_chat_fast",
    "call_with_fallback", "call_fast_with_fallback",
    "call_gemini_then_groq", "call_for_ingest",
    "get_groq_usage_report", "get_gemini_usage_report", "current_model", "GROQ_API_KEYS",
    "GEMINI_API_KEYS", "get_next_gemini_key", "mark_gemini_success", "mark_gemini_rate_limited",
    "mark_gemini_auth_failed",
]