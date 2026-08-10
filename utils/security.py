"""Login lockout, per-session rate limiting, and request fingerprinting.
Pulled out of app.py so admin.py and chat.py can both use it without
circular imports.
"""
import hashlib
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Tuple

from flask import request, session

LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300
LOGIN_LOCKOUT_SECONDS = 900

_login_attempts: Dict[str, List[float]] = defaultdict(list)
_login_lock = Lock()

_session_limits: Dict[str, List[float]] = defaultdict(list)
_session_lock = Lock()
SESSION_WINDOW_SECONDS = 300


def client_ip() -> str:
    return (request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
            or request.remote_addr or 'unknown')


def is_locked_out(ip: str) -> Tuple[bool, int]:
    now = time.time()
    with _login_lock:
        attempts = [t for t in _login_attempts[ip] if now - t < LOGIN_LOCKOUT_SECONDS]
        _login_attempts[ip] = attempts
        recent = [t for t in attempts if now - t < LOGIN_WINDOW_SECONDS]
        if len(recent) >= LOGIN_MAX_ATTEMPTS:
            remaining = int(LOGIN_LOCKOUT_SECONDS - (now - min(recent)))
            return True, max(remaining, 0)
    return False, 0


def record_failed_attempt(ip: str):
    with _login_lock:
        _login_attempts[ip].append(time.time())


def clear_attempts(ip: str):
    with _login_lock:
        _login_attempts.pop(ip, None)


def attempts_left(ip: str) -> int:
    now = time.time()
    return max(0, LOGIN_MAX_ATTEMPTS - sum(
        1 for t in _login_attempts[ip] if now - t < LOGIN_WINDOW_SECONDS))


def fingerprint() -> str:
    raw = client_ip() + request.headers.get('User-Agent', '')
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def admin_required() -> bool:
    if not session.get('admin'):
        return False
    if session.get('admin_ip') and session['admin_ip'] != client_ip():
        session.clear()
        return False
    return True


def check_session_rate_limit(session_id: str, max_messages: int) -> Tuple[bool, int]:
    now = time.time()
    with _session_lock:
        reqs = [t for t in _session_limits[session_id] if now - t < SESSION_WINDOW_SECONDS]
        _session_limits[session_id] = reqs
        if len(reqs) >= max_messages:
            return False, len(reqs)
        _session_limits[session_id].append(now)
        return True, len(reqs) + 1


def cleanup_old_sessions():
    with _session_lock:
        now = time.time()
        expired = [sid for sid, ts in _session_limits.items() if now - max(ts) > 604800]
        for sid in expired:
            del _session_limits[sid]
