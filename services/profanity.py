import re
from typing import List


PROFANITY_TERMS = (
    "putang ina", "putangina", "tangina", "taena", "pota", "puta",
    "tngina", "ptngina",
    "gago", "gaga", "g@go", "gag0",
    "tanga", "t@nga", "tang@", "tnga",
    "bobo", "boba", "b0b0", "vovo", "obob",
    "ulol", "ul0l", "ulul",
    "tarantado", "tarantada",
    "leche", "letse",
    "punyeta", "pnyeta", "pakshet", "paksit",
    "hudas",
    "lintik",
    "kantot", "kntot", "kantutan",
    "titi", "tite", "etits", "burat",
    "puke", "kepyas", "bilat",
    "jakol", "jabol", "j@kol",
    "tamod",
    "pokpok",
    "malibog",
    "chupa", "tsupa",
    "kupal",
    "hayop ka", "hayup ka",
    "demonyo", "demoyo",
    "peste",
    "bwisit", "buwisit",
    "ungas",
    "engot",
    "balasubas",
    "salot",
    "hampaslupa",
    "patay-gutom", "patay gutom",
    "mukhang pera",
    "amats",
    "epal",
    "kanser",
    "bano",
    "sipsip",
    "plastik",
    "dugyot",
    "squammy", "skwater",
    "bonak",
    "shunga",
    "yawa",
    "piste", "pisteng yawa",
    "atay", "gi-atay", "giatay",
    "buang", "boang",
    "animal",
    "ukininam",
    "takneng",
    "esmeringhoy",
)


def _term_pattern(term: str) -> re.Pattern:
    parts = re.split(r"[\s-]+", term)
    body = r"[\W_]+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


_PROFANITY_PATTERNS = tuple(
    (term, _term_pattern(term))
    for term in sorted(set(PROFANITY_TERMS), key=len, reverse=True)
)


def find_flagged_terms(text: str) -> List[str]:
    """Return listed terms found as complete words or phrases."""
    if not text:
        return []
    return [
        term
        for term, pattern in _PROFANITY_PATTERNS
        if pattern.search(text)
    ]
