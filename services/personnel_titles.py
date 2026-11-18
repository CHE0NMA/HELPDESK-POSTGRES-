import re
from typing import Dict, FrozenSet


_ROLE_ALIASES = {
    "college_president": (
        "college president",
        "kinsa ang college president",
        "president of the college",
        "president of college",
        "school president",
        "president of the school",
        "presidente sa college",
        "presidente sa eskwelahan",
        "presidente sa kolehiyo",
        "pangulo sa eskwelahan",
        "pangulo sa kolehiyo",
        "sino ang presidente ng paaralan",
        "sino ang presidente ng kolehiyo",
        "sino ang pangulo ng paaralan",
        "sino ang pangulo ng kolehiyo",
    ),
    "vp_finance": (
        "vp finance",
        "vp of finance",
        "bise presidente sa finance",
        "bise presidente sa finance",
        "bise presidente sa pinansya",
        "bise presidente ng pananalapi",
        "pangalawang pangulo ng pananalapi",
        "vice president finance",
        "vice president for finance",
        "vice president of finance",
    ),
    "vp_academic_affairs": (
        "vp academic affairs",
        "vp of academic affairs",
        "bise presidente sa academic affairs",
        "bise presidente sa academic affairs",
        "bise presidente sa akademikong kalihokan",
        "bise presidente ng akademikong gawain",
        "pangalawang pangulo ng akademikong gawain",
        "vice president academic affairs",
        "vice president for academic affairs",
        "vice president of academic affairs",
    ),
    "vp_student_affairs_services_hrmo": (
        "vp student affairs services hrmo",
        "vp student affairs and services hrmo",
        "vp student affairs and services",
        "vp of student affairs and services",
        "bise presidente sa student affairs",
        "bise presidente sa student affairs and services",
        "bise presidente sa hrmo",
        "bise presidente sa serbisyo sa estudyante",
        "bise presidente ng ugnayang pang estudyante",
        "pangalawang pangulo ng ugnayang pang estudyante",
        "vice president student affairs and services",
        "vice president for student affairs and services",
        "vice president of student affairs and services",
        "hrmo",
    ),
    "vp_development": (
        "vp development",
        "vp of development",
        "bise presidente sa development",
        "bise presidente sa development",
        "bise presidente sa kalamboan",
        "bise presidente ng pagpapaunlad",
        "pangalawang pangulo ng pagpapaunlad",
        "vice president development",
        "vice president for development",
        "vice president of development",
    ),
    "treasurer": (
        "treasurer",
        "college treasurer",
        "tig tipig sa kwarta",
        "tigtipig sa kwarta",
        "tresurero sa eskwelahan",
        "ingatyaman ng paaralan",
        "ingat yaman ng paaralan",
        "ingatyaman ng kolehiyo",
        "ingat yaman ng kolehiyo",
    ),
    "accountant": (
        "accountant",
        "college accountant",
        "tig kwenta sa eskwelahan",
        "tigkwenta sa eskwelahan",
        "akawntant sa eskwelahan",
        "kuwentador ng paaralan",
        "accountant ng paaralan",
    ),
    "director_student_finance": (
        "director student finance",
        "director of student finance",
        "student finance director",
        "direktor sa student finance",
        "direktor sa student finance",
        "direktor sa panalapi sa estudyante",
        "direktor ng pananalapi ng estudyante",
    ),
    "dean_computer_studies": (
        "dean computer studies",
        "dean sa computer studies",
        "dean ng computer studies",
        "dean of the computer studies",
        "dean of computer studies",
        "dean college computer studies",
        "dean of college computer studies",
        "dean of the college of computer studies",
        "dean sa college of computer studies",
        "dean ng college of computer studies",
        "dean computer of studies",
        "computer studies dean",
        "dean ng computer studies",
        "dean ng kolehiyo ng computer studies",
    ),
    "bscs_coordinator": (
        "bscs coordinator",
        "coordinator sa bscs",
        "coordinator sa computer science",
        "kinsa ang coordinator sa bscs",
        "coordinator ng bscs",
        "coordinator ng computer science",
        "coordinator bscs",
        "computer science coordinator",
        "coordinator of computer science",
        "coordinator ng computer science",
    ),
    "bsit_coordinator": (
        "bsit coordinator",
        "coordinator sa bsit",
        "coordinator sa information technology",
        "kinsa ang coordinator sa bsit",
        "coordinator ng bsit",
        "coordinator ng information technology",
        "coordinator bsit",
        "information technology coordinator",
        "coordinator of information technology",
        "coordinator ng information technology",
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
