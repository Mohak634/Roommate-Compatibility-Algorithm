"""DormMate matching engine: weighted similarity over lifestyle answers.

A profile is a dict of survey answers keyed by question (the same keys the
survey schema and the original CSVs use). Each factor compares two people's
answers and gives a similarity between 0 and 1; the match score is the
weighted average of those similarities, as a percentage.

Deliberately NOT inputs:
- "Roommate Compatibility Rating": a rating of an existing roommate pair. It is
  the ground truth the engine is evaluated against (scripts/evaluate_matching.py),
  so using it as an input would leak the answer.
- "Mutual Friends": how many friends someone shares with their current roommate,
  a property of an existing pair, not of the person.
- Age and gender: filters, not compatibility factors.

Scores are absolute (the same pair always gets the same score), symmetric
(A vs B == B vs A) and never rescaled per list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from app.core.vocab import HOBBY_OPTIONS, ORDINAL_MAPPINGS

# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

_COURSE_ALIASES = {
    "btech": "BTech", "b.tech": "BTech", "b tech": "BTech", "computer science": "BTech",
    "cse": "BTech", "engineering": "BTech",
    "bsc": "BSc", "b.sc": "BSc", "b sc": "BSc",
    "ba": "BA", "b.a": "BA",
    "bcom": "BCom", "b.com": "BCom",
    "bba": "BBA", "psychology": "Psychology",
}


def _clean(value) -> str | None:
    if value is None:
        return None
    s = " ".join(str(value).split())
    return None if s == "" or s.lower() in ("nan", "none", "empty", "unknown") else s


def normalise_course(value) -> str | None:
    s = _clean(value)
    if s is None:
        return None
    key = s.lower().rstrip(".")
    return _COURSE_ALIASES.get(key, s.title())


def _ordinal(question: str, value) -> int | None:
    s = _clean(value)
    if s is None:
        return None
    for label, code in ORDINAL_MAPPINGS[question].items():
        if label.lower() == s.lower():
            return code
    return None


def _scale(value) -> float | None:
    s = _clean(value)
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def hobbies_of(profile: dict) -> set[str]:
    """Hobbies from either a 'Hobbies' list (app) or 'Hobby 1..3' columns (CSV)."""
    raw: Iterable = profile.get("Hobbies") or [profile.get(f"Hobby {i}") for i in (1, 2, 3)]
    if isinstance(raw, str):
        raw = raw.split(",")
    canon = {h.lower(): h for h in HOBBY_OPTIONS if h != "Empty"}
    out = set()
    for h in raw:
        s = _clean(h)
        if s and s.lower() in canon:
            out.add(canon[s.lower()])
    return out


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Factor:
    key: str          # question key in the profile
    label: str        # short label shown in the app
    group: str        # factor group (from the original question mapping)
    weight: float     # hand-set weight (from the original question mapping)
    similarity: Callable[[dict, dict], float | None]


def _scale_sim(key: str, lo: float = 1, hi: float = 10):
    def sim(a: dict, b: dict):
        x, y = _scale(a.get(key)), _scale(b.get(key))
        if x is None or y is None:
            return None
        return 1 - abs(x - y) / (hi - lo)
    return sim


def _ordinal_sim(key: str):
    span = max(ORDINAL_MAPPINGS[key].values()) - min(ORDINAL_MAPPINGS[key].values())

    def sim(a: dict, b: dict):
        x, y = _ordinal(key, a.get(key)), _ordinal(key, b.get(key))
        if x is None or y is None:
            return None
        return 1 - abs(x - y) / span
    return sim


def _same_sim(key: str, norm=_clean):
    def sim(a: dict, b: dict):
        x, y = norm(a.get(key)), norm(b.get(key))
        if x is None or y is None:
            return None
        return 1.0 if x.lower() == y.lower() else 0.0
    return sim


def _hobby_sim(a: dict, b: dict):
    x, y = hobbies_of(a), hobbies_of(b)
    if not x or not y:
        return None
    return len(x & y) / len(x | y)


def _year_sim(a: dict, b: dict):
    x, y = _ordinal("Year of Study", a.get("Year of Study")), _ordinal("Year of Study", b.get("Year of Study"))
    if x is None or y is None:
        return None
    return max(0.0, 1 - abs(x - y) / 3)


# Weights are the original hand-set question-mapping weights.
FACTORS: list[Factor] = [
    Factor("Cleanliness Scale", "Cleanliness", "Cleanliness", 0.129, _scale_sim("Cleanliness Scale")),
    Factor("Cleaning Frequency", "Cleaning habits", "Cleanliness", 0.129, _ordinal_sim("Cleaning Frequency")),
    Factor("Bedtime", "Bedtime", "Sleep", 0.115, _ordinal_sim("Bedtime")),
    Factor("Noise Tolerance Sleep", "Noise while sleeping", "Sleep", 0.115, _scale_sim("Noise Tolerance Sleep")),
    Factor("Study Frequency", "Study routine", "Study", 0.109, _ordinal_sim("Study Frequency")),
    Factor("Study Location", "Study spot", "Study", 0.109, _same_sim("Study Location")),
    Factor("Noise Tolerance Study", "Noise while studying", "Study", 0.109, _scale_sim("Noise Tolerance Study")),
    Factor("Social Preference", "Social energy", "Social", 0.112, _ordinal_sim("Social Preference")),
    Factor("Roommate Host", "Hosting friends", "Social", 0.112, _same_sim("Roommate Host")),
    Factor("Party Frequency", "Parties", "Social", 0.114, _ordinal_sim("Party Frequency")),
    Factor("Hobbies", "Shared hobbies", "Interests", 0.113, _hobby_sim),
    Factor("Personal Space", "Personal space", "Space", 0.121, _ordinal_sim("Personal Space")),
    Factor("Year of Study", "Year of study", "Academics", 0.108, _year_sim),
    Factor("Course of Study", "Course", "Academics", 0.108, _same_sim("Course of Study", normalise_course)),
]

FACTOR_KEYS = [f.key for f in FACTORS]


def _load_production_weights() -> dict[str, float]:
    """Weights validated against real roommate pairs (app/core/weights.json, written by
    scripts/evaluate_matching.py). Falls back to the hand-set weights if the file is missing."""
    import json
    from pathlib import Path

    path = Path(__file__).with_name("weights.json")
    try:
        weights = json.loads(path.read_text())["weights"]
        if set(weights) == set(FACTOR_KEYS):
            return weights
    except (OSError, ValueError, KeyError):
        pass
    return {f.key: f.weight for f in FACTORS}


WEIGHTS: dict[str, float] = _load_production_weights()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def factor_similarities(a: dict, b: dict) -> dict[str, float]:
    """Similarity (0..1) per factor; factors either person left blank are omitted."""
    out = {}
    for f in FACTORS:
        s = f.similarity(a, b)
        if s is not None:
            out[f.key] = s
    return out


def score(a: dict, b: dict, weights: dict[str, float] | None = None) -> float:
    """Compatibility score 0..100: weighted mean of factor similarities."""
    sims = factor_similarities(a, b)
    w = weights or WEIGHTS
    total = sum(w[k] for k in sims)
    if total == 0:
        return 0.0
    return 100 * sum(w[k] * s for k, s in sims.items()) / total


def explain(a: dict, b: dict, k: int = 3) -> dict:
    """The factors two people agree on most and differ on most."""
    labels = {f.key: f.label for f in FACTORS}
    sims = factor_similarities(a, b)
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    closest = [labels[key] for key, s in ranked if s >= 0.75][:k]
    differences = [labels[key] for key, s in reversed(ranked) if s <= 0.5][:k]
    return {"closest": closest, "differences": differences}


def breakdown(a: dict, b: dict) -> list[dict]:
    """Per-factor similarity as percentages, for the match-detail chart."""
    sims = factor_similarities(a, b)
    return [{"factor": f.label, "similarity": round(100 * sims[f.key], 1)} for f in FACTORS if f.key in sims]


def rank(reference: dict, candidates: Iterable[dict], top_n: int = 10, id_key: str = "id",
         weights: dict[str, float] | None = None) -> list[dict]:
    """Top matches for a reference profile, best first. Excludes the reference itself."""
    ref_id = reference.get(id_key)
    scored = []
    for c in candidates:
        if ref_id is not None and c.get(id_key) == ref_id:
            continue
        scored.append((score(reference, c, weights), c))
    scored.sort(key=lambda t: (-t[0], str(t[1].get(id_key))))
    return [
        {"id": c.get(id_key), "score": round(s, 1), **explain(reference, c)}
        for s, c in scored[:top_n]
    ]
