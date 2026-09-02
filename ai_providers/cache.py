"""TTL caches used across the app.

AICache: generic (prompt-ish args) -> AI result cache, used by dates/faq/handbook.
FAQCache: normalized-question -> final rendered answer cache used by routes/chat.py.
Both get automatic cleanup instead of relying on maxsize eviction alone,
addressing the "no TTL-based cache invalidation" review item.
"""
import hashlib
import time
from threading import Lock
from typing import Any, Dict, Optional

from .metrics import model_metrics


class AICache:
    def __init__(self, maxsize: int = 500, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple] = {}
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.lock = Lock()
        self._last_cleanup = time.time()

    def _key(self, *args) -> str:
        return hashlib.md5("|".join(str(a).lower() for a in args).encode()).hexdigest()

    def _cleanup_if_due(self):
        if time.time() - self._last_cleanup < 300:  # every 5 min
            return
        now = time.time()
        with self.lock:
            expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
            for k in expired:
                del self.cache[k]
            self._last_cleanup = now

    def get(self, *args) -> Optional[Any]:
        self._cleanup_if_due()
        k = self._key(*args)
        with self.lock:
            if k in self.cache:
                val, ts = self.cache[k]
                if time.time() - ts < self.ttl:
                    model_metrics.record_cache_hit()
                    return val
                del self.cache[k]
        model_metrics.record_cache_miss()
        return None

    def set(self, *args, result=None):
        if result is None and len(args) > 0:
            *key_parts, result = args
            args = key_parts
        k = self._key(*args)
        with self.lock:
            if len(self.cache) >= self.maxsize:
                oldest = min(self.cache.keys(), key=lambda x: self.cache[x][1])
                del self.cache[oldest]
            self.cache[k] = (result, time.time())

    def invalidate(self):
        with self.lock:
            self.cache.clear()


class FAQCache:
    # BUG FIX (perf review): 100 was small relative to AICache's 500 --
    # under normal traffic (and especially during enrollment-period
    # spikes) this evicted and forced full re-computation of FAQ answers
    # far more often than necessary. Raised to match AICache so a busy
    # period doesn't quietly degrade into a cache-miss-most-of-the-time
    # state. ttl_seconds unchanged -- eviction pressure, not staleness,
    # was the actual problem.
    def __init__(self, maxsize=500, ttl_seconds=3600):
        self.cache = {}
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.lock = Lock()

    def _key(self, q):
        return hashlib.md5(q.encode()).hexdigest()

    def get(self, q):
        k = self._key(q)
        with self.lock:
            if k in self.cache:
                e = self.cache[k]
                if time.time() - e['ts'] < self.ttl:
                    return e['data']
                del self.cache[k]
        return None

    def set(self, q, data):
        k = self._key(q)
        with self.lock:
            if len(self.cache) >= self.maxsize and k not in self.cache:
                oldest = min(self.cache, key=lambda x: self.cache[x]['ts'])
                del self.cache[oldest]
            # BUG FIX: invalidate_pattern() used to substring-match against
            # `self.cache`'s own keys, but those keys are MD5 hex digests
            # (see _key()) -- a pattern like "today" or "registrar" can
            # NEVER appear in a hex string (0-9a-f only), so
            # invalidate_pattern() was a permanent no-op. Keeping the raw
            # question text alongside the hash gives it something real to
            # match against.
            self.cache[k] = {'data': data, 'ts': time.time(), 'raw_key': q}

    def invalidate(self):
        with self.lock:
            self.cache.clear()

    def invalidate_pattern(self, pattern):
        pattern = pattern.lower()
        with self.lock:
            for k in [k for k, e in self.cache.items()
                      if pattern in e.get('raw_key', '').lower()]:
                del self.cache[k]


ai_cache = AICache()
faq_cache = FAQCache()