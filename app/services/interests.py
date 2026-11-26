"""Likes and mutual matches.

Collection: interests { from_user, to_user, action: "like" | "skip", score, created_at, demo_auto }

Sample profiles can't respond, so a like on a sample profile whose server-side
score is at least AUTO_LIKE_SCORE is answered automatically (flagged
demo_auto). Likes between real users need both people.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException

from app.core import matching
from app.services import profile_store
from app.services.db import get_db

AUTO_LIKE_SCORE = 70.0


def _db():
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    return db


def is_mutual(username: str, profile_id: str) -> bool:
    db = _db()
    return bool(db.interests.find_one({"from_user": username, "to_user": profile_id, "action": "like"})
                and db.interests.find_one({"from_user": profile_id, "to_user": username, "action": "like"}))


def record(username: str, profile_id: str, action: str) -> dict:
    if action not in ("like", "skip"):
        raise HTTPException(status_code=422, detail="action must be 'like' or 'skip'.")
    if profile_id == username:
        raise HTTPException(status_code=400, detail="You can't like yourself.")
    me, other = profile_store.get_profile(username), profile_store.get_profile(profile_id)
    if me is None:
        raise HTTPException(status_code=409, detail="Finish the survey first.")
    if other is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    db, now = _db(), datetime.now(timezone.utc)
    s = round(matching.score(me["answers"], other["answers"]), 1)
    db.interests.replace_one({"from_user": username, "to_user": profile_id},
                             {"from_user": username, "to_user": profile_id, "action": action,
                              "score": s, "created_at": now}, upsert=True)
    if action == "like" and other.get("is_sample") and s >= AUTO_LIKE_SCORE:
        db.interests.replace_one({"from_user": profile_id, "to_user": username},
                                 {"from_user": profile_id, "to_user": username, "action": "like",
                                  "score": s, "demo_auto": True, "created_at": now}, upsert=True)
    return {"mutual": action == "like" and is_mutual(username, profile_id), "score": s}


def sent(username: str) -> list[dict]:
    db = _db()
    likes = list(db.interests.find({"from_user": username, "action": "like"}))
    back = {d["from_user"] for d in db.interests.find(
        {"from_user": {"$in": [d["to_user"] for d in likes]}, "to_user": username, "action": "like"})}
    out = []
    for d in sorted(likes, key=lambda d: d["created_at"], reverse=True):
        p = profile_store.get_profile(d["to_user"])
        if p:
            out.append({"id": p["id"], "name": p.get("name"), "photo_url": p.get("photo_url"),
                        "score": d.get("score"), "mutual": d["to_user"] in back})
    return out


def mutual(username: str) -> list[dict]:
    return [e for e in sent(username) if e["mutual"]]
