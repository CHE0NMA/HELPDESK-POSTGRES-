"""Circuit breaker for AI provider calls.

Its own module (instead of a global in app.py) so any service — dates,
handbook, faq — can check circuit_breaker.is_open() without importing the
whole Flask app.
"""
import time
from threading import Lock


class AICircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failures = 0
        self.last_failure = 0
        self.threshold = failure_threshold
        self.timeout = recovery_timeout
        self.lock = Lock()
        self.half_open = False
        self.half_open_attempts = 0

    def is_open(self) -> bool:
        with self.lock:
            if self.half_open:
                return False
            if self.failures >= self.threshold:
                if time.time() - self.last_failure > self.timeout:
                    self.half_open = True
                    self.half_open_attempts = 0
                    return False
                return True
            return False

    def record_success(self):
        with self.lock:
            self.failures = 0
            self.half_open = False

    def record_failure(self):
        with self.lock:
            if self.half_open:
                self.half_open_attempts += 1
                if self.half_open_attempts >= 2:
                    self.half_open = False
                    self.failures = self.threshold + 1
                    self.last_failure = time.time()
            else:
                self.failures += 1
                self.last_failure = time.time()

    def get_state(self) -> dict:
        with self.lock:
            return {
                'is_open': self.failures >= self.threshold and not self.half_open,
                'failures': self.failures,
                'threshold': self.threshold,
                'half_open': self.half_open,
                'time_since_last_failure': time.time() - self.last_failure if self.last_failure else None,
            }


circuit_breaker = AICircuitBreaker()
