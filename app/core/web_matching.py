"""JSON responses for the matching endpoints, built on app.core.matching.

Response shapes match what the Flutter app already reads (name, score,
top_factors, course, radar, plus a few profile fields), with two additions:
`differences` (where the two people disagree most) and `breakdown`
(per-factor similarity for the match-detail chart).
"""

from __future__ import annotations

import re

from app.core import matching
from app.core.matching import normalise_course
from app.core.vocab import ORDINAL_MAPPINGS
from app.services import profile_store

METHOD = "weighted-similarity"

# Radar axes: one per lifestyle question, each scaled to 0..100 from the
# answer options so "you" and "them" are comparable.
_RADAR = [
    ("Bedtime", "Bedtime", "ordinal"),
    ("Cleaning Frequency", "Cleaning habits", "ordinal"),
    ("Cleanliness Scale", "Cleanliness", "scale"),
    ("Year of Study", "Year of study", "ordinal"),
    ("Party Frequency", "Parties", "ordinal"),
    ("Personal Space", "Personal space", "ordinal"),
    ("Social Preference", "Social energy", "ordinal"),
    ("Noise Tolerance Study", "Noise while studying", "scale"),
    ("Study Frequency", "Study routine", "ordinal"),
    ("Noise Tolerance Sleep", "Noise while sleeping", "scale"),
]


def _axis_value(profile: dict, key: str, kind: str) -> float | None:
    raw = profile.get(key)
    if kind == "scale":
        try:
            return round((float(raw) - 1) / 9 * 100, 1)
        except (TypeError, ValueError):
            return None
    code = matching._ordinal(key, raw)
    if code is None:
        return None
    lo, hi = min(ORDINAL_MAPPINGS[key].values()), max(ORDINAL_MAPPINGS[key].values())
    return round((code - lo) / (hi - lo) * 100, 1)


def _radar(a: dict, b: dict) -> list[dict]:
    out = []
    for key, label, kind in _RADAR:
        you, them = _axis_value(a, key, kind), _axis_value(b, key, kind)
        if you is not None and them is not None:
            out.append({"label": label, "you": you, "them": them})
    return out


def _extras(p: dict) -> dict:
    extras = {}
    for key in ("Age", "Cleanliness Scale"):
        try:
            extras[key] = int(float(p.get(key)))
        except (TypeError, ValueError):
            extras[key] = None
    extras["Year of Study"] = matching._ordinal("Year of Study", p.get("Year of Study"))
    return extras


def _student_email(name):
    """MOCK contact detail derived from the name; the dataset has no emails."""
    slug = re.sub(r"[^a-z0-9]+", ".", str(name).lower()).strip(".")
    return f"{slug}@dormmate.edu"


def list_users() -> list[str]:
    return sorted(p["id"] for p in profile_store.all_profiles())


def get_matches(reference_name, top_n=10, **_ignored):
    ref = profile_store.get_profile_by_id(reference_name)
    if ref is None:
        return {"success": False, "message": "User not found in dataset.", "matches": []}
    by_id = {p["id"]: p for p in profile_store.all_profiles()}
    ranked = matching.rank(ref, by_id.values(), top_n=top_n)
    matches = []
    for m in ranked:
        cand = by_id[m["id"]]
        matches.append({
            "name": m["id"],
            "score": m["score"],
            "top_factors": m["closest"],
            "differences": m["differences"],
            "course": normalise_course(cand.get("Course of Study")),
            "radar": _radar(ref, cand),
            "breakdown": matching.breakdown(ref, cand),
            **_extras(cand),
        })
    return {"success": True, "reference": reference_name, "method": METHOD, "matches": matches}


def compare_users_json(user1_name, user2_name, **_ignored):
    a, b = profile_store.get_profile_by_id(user1_name), profile_store.get_profile_by_id(user2_name)
    for name, p in ((user1_name, a), (user2_name, b)):
        if p is None:
            return {"success": False, "message": f"'{name}' not found in dataset."}
    why = matching.explain(a, b)
    return {
        "success": True,
        "user1": user1_name,
        "user2": user2_name,
        "method": METHOD,
        "compatibility_score": round(matching.score(a, b), 1),
        "top_factors": why["closest"],
        "differences": why["differences"],
        "radar": _radar(a, b),
        "breakdown": matching.breakdown(a, b),
        "course": normalise_course(b.get("Course of Study")),
        "student_email": _student_email(user2_name),
        **_extras(b),
    }
