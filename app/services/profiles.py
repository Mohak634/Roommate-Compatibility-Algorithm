"""Validate and save a user's survey answers."""

from __future__ import annotations

from fastapi import HTTPException

from app.services import profile_store
from app.services.survey import get_survey_schema


def validate(answers: dict) -> list[str]:
    """Problems with the answers against the survey schema (empty list if fine)."""
    questions = get_survey_schema()["questions"]
    schema = {q["key"]: q for q in questions}
    problems = [f"Missing answer: {q['label']}" for q in questions if q["key"] not in answers]
    for key, value in answers.items():
        q = schema.get(key)
        if q is None:
            problems.append(f"Unknown question '{key}'")
            continue
        t = q["type"]
        if t in ("scale", "number"):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"'{key}' must be a number")
            elif not (q.get("min", float("-inf")) <= value <= q.get("max", float("inf"))):
                problems.append(f"'{key}' out of range")
        elif t == "choice":
            if value not in q["options"]:
                problems.append(f"'{key}' has invalid option '{value}'")
        elif t == "multi":
            if not isinstance(value, list) or not value:
                problems.append(f"'{key}' must be a non-empty list")
            elif len(value) > q.get("maxSelections", len(q["options"])):
                problems.append(f"'{key}' allows at most {q['maxSelections']} selections")
            elif any(v not in q["options"] for v in value):
                problems.append(f"'{key}' contains an invalid option")
        elif t == "text":
            if not isinstance(value, str) or not value.strip():
                problems.append(f"'{key}' must be a non-empty string")
    return problems


def save(username: str, answers: dict) -> dict:
    problems = validate(answers)
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems))
    return profile_store.save_user_profile(username, answers)
