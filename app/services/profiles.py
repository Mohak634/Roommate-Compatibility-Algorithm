"""User survey answers, stored in the (pre-existing) dormmate.preferences
collection in the survey's raw vocabulary so they encode identically to the
Google-Forms training data.

Doc shape: { username, answers: {<question key>: <raw answer>}, updated_at }
Saving also flips users.profile_complete to True.
"""

from datetime import datetime, timezone

from app.services.mongo import connect_to_mongo
from app.services.survey import get_survey_schema


def _validate(answers):
    """Light validation against the schema. Returns a list of problems."""
    schema = {q["key"]: q for q in get_survey_schema()["questions"]}
    problems = []
    for key, value in answers.items():
        q = schema.get(key)
        if q is None:
            problems.append(f"Unknown question '{key}'")
            continue
        t = q["type"]
        if t in ("scale", "number"):
            if not isinstance(value, (int, float)):
                problems.append(f"'{key}' must be a number")
            elif not (q.get("min", float("-inf")) <= value <= q.get("max", float("inf"))):
                problems.append(f"'{key}' out of range")
        elif t == "choice":
            if value not in q["options"]:
                problems.append(f"'{key}' has invalid option '{value}'")
        elif t == "multi":
            if not isinstance(value, list):
                problems.append(f"'{key}' must be a list")
            elif len(value) > q.get("maxSelections", len(q["options"])):
                problems.append(f"'{key}' allows at most {q['maxSelections']} selections")
            elif any(v not in q["options"] for v in value):
                problems.append(f"'{key}' contains an invalid option")
        elif t == "text":
            if not isinstance(value, str) or not value.strip():
                problems.append(f"'{key}' must be a non-empty string")
    return problems


def save_profile(username, answers):
    problems = _validate(answers)
    if problems:
        return {"success": False, "message": "; ".join(problems)}

    client = connect_to_mongo()
    if not client:
        return {"success": False, "message": "MongoDB connection failed."}
    try:
        db = client["dormmate"]
        db["preferences"].replace_one(
            {"username": username},
            {
                "username": username,
                "answers": answers,
                "updated_at": datetime.now(timezone.utc),
            },
            upsert=True,
        )
        db["users"].update_one(
            {"username": username}, {"$set": {"profile_complete": True}}
        )
        return {"success": True, "message": "Profile saved."}
    finally:
        client.close()


def get_profile(username):
    client = connect_to_mongo()
    if not client:
        return {"success": False, "message": "MongoDB connection failed."}
    try:
        doc = client["dormmate"]["preferences"].find_one({"username": username})
        if not doc:
            return {"success": True, "answers": None}
        return {"success": True, "answers": doc.get("answers", {})}
    finally:
        client.close()
