"""Match lists and match details for the logged-in user."""

from __future__ import annotations

import re

from fastapi import HTTPException

from app.core import matching
from app.core.matching import normalise_course
from app.core.vocab import ORDINAL_MAPPINGS
from app.services import profile_store

# Radar axes: one per lifestyle question, scaled to 0..100 from the answer options.
_RADAR = [
    ("Bedtime", "Bedtime", "ordinal"),
    ("Cleaning Frequency", "Cleaning habits", "ordinal"),
    ("Cleanliness Scale", "Cleanliness", "scale"),
    ("Party Frequency", "Parties", "ordinal"),
    ("Personal Space", "Personal space", "ordinal"),
    ("Social Preference", "Social energy", "ordinal"),
    ("Noise Tolerance Study", "Noise while studying", "scale"),
    ("Study Frequency", "Study routine", "ordinal"),
    ("Noise Tolerance Sleep", "Noise while sleeping", "scale"),
]


def _axis(answers: dict, key: str, kind: str) -> float | None:
    raw = answers.get(key)
    if kind == "scale":
        v = matching._scale(raw)
        return None if v is None else round((v - 1) / 9 * 100, 1)
    code = matching._ordinal(key, raw)
    if code is None:
        return None
    values = ORDINAL_MAPPINGS[key].values()
    return round((code - min(values)) / (max(values) - min(values)) * 100, 1)


def radar(a: dict, b: dict) -> list[dict]:
    out = []
    for key, label, kind in _RADAR:
        you, them = _axis(a, key, kind), _axis(b, key, kind)
        if you is not None and them is not None:
            out.append({"label": label, "you": you, "them": them})
    return out


def card(p: dict) -> dict:
    a = p.get("answers", {})
    age = a.get("Age")
    return {
        "id": p["id"],
        "name": p.get("name"),
        "is_sample": bool(p.get("is_sample")),
        "photo_url": p.get("photo_url"),
        "bio": p.get("bio"),
        "lifestyle_type": p.get("lifestyle_type"),
        "course": normalise_course(a.get("Course of Study")),
        "year": a.get("Year of Study"),
        "age": int(age) if isinstance(age, (int, float)) else None,
        "hobbies": sorted(matching.hobbies_of(a)),
    }


def my_profile(username: str) -> dict:
    me = profile_store.get_profile(username)
    if me is None:
        raise HTTPException(status_code=409, detail="Finish the survey to see matches.")
    return me


def list_matches(username: str, top_n: int = 10) -> dict:
    me = my_profile(username)
    candidates = [p for p in profile_store.all_profiles() if p["id"] != username]
    by_id = {p["id"]: p for p in candidates}
    answers = [dict(p["answers"], id=p["id"]) for p in candidates]
    ranked = matching.rank(dict(me["answers"], id=username), answers, top_n=top_n)
    return {
        "you": card(me),
        "method": "weighted-similarity",
        "matches": [{**card(by_id[m["id"]]), "score": m["score"], "closest": m["closest"],
                     "differences": m["differences"]} for m in ranked],
    }


def _contact(p: dict, db_user_email: str | None) -> str:
    if not p.get("is_sample") and db_user_email:
        return db_user_email
    slug = re.sub(r"[^a-z0-9]+", ".", str(p.get("name", p["id"])).lower()).strip(".")
    return f"{slug}@dormmate.example"


def match_detail(username: str, profile_id: str, is_mutual: bool, other_email: str | None = None) -> dict:
    if profile_id == username:
        raise HTTPException(status_code=400, detail="That's you.")
    me = my_profile(username)
    other = profile_store.get_profile(profile_id)
    if other is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    a, b = me["answers"], other["answers"]
    why = matching.explain(a, b)
    return {
        **card(other),
        "score": round(matching.score(a, b), 1),
        "closest": why["closest"],
        "differences": why["differences"],
        "radar": radar(a, b),
        "breakdown": matching.breakdown(a, b),
        "mutual": is_mutual,
        "contact_email": _contact(other, other_email) if is_mutual else None,
    }
