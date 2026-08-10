"""Response-time / success-rate tracking for AI calls, separated from
AICache so the two concerns (metrics vs. memoized answers) don't get
tangled the way they were in the original AICache class."""
import time
from threading import Lock
from typing import List
import numpy as np


class ModelMetrics:
    def __init__(self):
        self.response_times: List[float] = []
        self.successful_calls: int = 0
        self.failed_calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._lock = Lock()

    def record_response_time(self, duration_ms: float, success: bool):
        with self._lock:
            self.response_times.append(duration_ms)
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        with self._lock:
            self.cache_misses += 1

    def get_stats(self) -> dict:
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
