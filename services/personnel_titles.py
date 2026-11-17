import re
from typing import Dict, FrozenSet


_ROLE_ALIASES = {
    "vp_finance": (
        "vp finance",
        "vp of finance",
        "vice president finance",
        "vice president for finance",
        "vice president of finance",
    ),
    "vp_academic_affairs": (
        "vp academic affairs",
        "vp of academic affairs",
        "vice president academic affairs",
        "vice president for academic affairs",
        "vice president of academic affairs",
    ),
    "vp_student_affairs_services_hrmo": (
        "vp student affairs services hrmo",
        "vp student affairs and services hrmo",
        "vp student affairs and services",
        "vp of student affairs and services",
        "vice president student affairs and services",
        "vice president for student affairs and services",
        "vice president of student affairs and services",
        "hrmo",
    ),
    "vp_development": (
        "vp development",
        "vp of development",
        "vice president development",
        "vice president for development",
        "vice president of development",
    ),
    "treasurer": ("treasurer",),
    "accountant": ("accountant",),
    "director_student_finance": (
        "director student finance",
        "director of student finance",
        "student finance director",
    ),
    "dean_computer_studies": (
        "dean computer studies",
        "dean of computer studies",
        "dean college computer studies",
        "dean of college computer studies",
        "dean of the college of computer studies",
        "dean computer of studies",
        "computer studies dean",
    ),
    "bscs_coordinator": (
        "bscs coordinator",
        "coordinator bscs",
        "computer science coordinator",
        "coordinator of computer science",
    ),
    "bsit_coordinator": (
        "bsit coordinator",
        "coordinator bsit",
        "information technology coordinator",
        "coordinator of information technology",
    ),
}


def _normalize_title_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


_NORMALIZED_ROLE_ALIASES: Dict[str, FrozenSet[str]] = {
    role: frozenset(_normalize_title_text(alias) for alias in aliases)
    for role, aliases in _ROLE_ALIASES.items()
}

def personnel_titles(text: str) -> FrozenSet[str]:
    """Return personnel roles explicitly mentioned in a text string."""
    normalized = f" {_normalize_title_text(text)} "
    if not normalized.strip():
        return frozenset()
    return frozenset(
        role
        for role, aliases in _NORMALIZED_ROLE_ALIASES.items()
        if any(f" {alias} " in normalized for alias in aliases)
    )


def has_personnel_title(text: str) -> bool:
    return bool(personnel_titles(text))
