"""Response-time / success-rate tracking for AI calls, separated from
AICache so the two concerns (metrics vs. memoized answers) don't get
tangled the way they were in the original AICache class."""
from threading import Lock
from typing import Any, Dict, List, Optional
import numpy as np


class ModelMetrics:
    def __init__(self):
        self.response_times: List[float] = []
        self.successful_calls: int = 0
        self.failed_calls: int = 0
        self.provider_calls: Dict[str, Dict[str, int]] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._lock = Lock()
        # End-to-end answer timing, separate from the raw-AI-call tracking
        # above. record_response_time() only ever fires for an actual Groq/
        # Gemini round-trip (ai_cache misses) -- it has no idea a question
        # was served instantly out of faq_cache or the DB-backed
        # answer_success_cache, which is most of what makes repeat traffic
        # fast. This tracks the FULL find_answer() latency for every chat
        # request and buckets it by how it was actually served.
        self.cache_path_times: List[float] = []
        self.computed_path_times: List[float] = []
        # Anything answered this fast almost certainly never made a real
        # AI provider call (those take hundreds of ms to multiple seconds);
        # generous on purpose so a fast local DB lookup still counts.
        self.ANSWER_CACHE_THRESHOLD_MS = 300.0

    def record_answer_timing(self, duration_ms: float) -> None:
        with self._lock:
            if duration_ms <= self.ANSWER_CACHE_THRESHOLD_MS:
                self.cache_path_times.append(duration_ms)
            else:
                self.computed_path_times.append(duration_ms)
            if len(self.cache_path_times) > 1000:
                self.cache_path_times = self.cache_path_times[-1000:]
            if len(self.computed_path_times) > 1000:
                self.computed_path_times = self.computed_path_times[-1000:]

    def get_answer_timing_stats(self) -> Dict[str, float | int]:
        with self._lock:
            cache_times = list(self.cache_path_times)
            db_times = list(self.computed_path_times)
        return {
            'avg_cache_ms': round(float(np.mean(cache_times)), 1) if cache_times else 0.0,
            'avg_db_ms': round(float(np.mean(db_times)), 1) if db_times else 0.0,
            'cache_hit_count': len(cache_times),
            'computed_count': len(db_times),
        }

    def record_response_time(
        self, duration_ms: float, success: bool, provider: Optional[str] = None
    ) -> None:
        with self._lock:
            self.response_times.append(duration_ms)
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
            if provider:
                calls = self.provider_calls.setdefault(
                    provider, {"successful_calls": 0, "failed_calls": 0}
                )
                calls["successful_calls" if success else "failed_calls"] += 1
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def get_provider_stats(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {
                name: dict(calls) for name, calls in self.provider_calls.items()
            }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.successful_calls + self.failed_calls
            avg = np.mean(self.response_times) if self.response_times else 0
            p95 = np.percentile(self.response_times, 95) if self.response_times else 0
            return {
                'successful_calls': self.successful_calls,
                'failed_calls': self.failed_calls,
                'success_rate': self.successful_calls / total if total > 0 else 0,
                'avg_response_time_ms': round(avg, 2),
                'p95_response_time_ms': round(p95, 2),
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0,
            }


model_metrics = ModelMetrics()