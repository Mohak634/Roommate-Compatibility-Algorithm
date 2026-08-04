"""Likes ("express interest") and mutual-match queries, backed by MongoDB.

Collection: dormmate.interests
  { from_user, to_user, action: "like" | "skip", score, created_at,
    demo_auto: bool  # present on auto-reciprocated likes (demo behavior) }

Demo behavior: dataset people can't swipe, so when a real user likes someone
with a compatibility score >= AUTO_RECIPROCATE_THRESHOLD, we insert the
reciprocal like automatically (flagged with demo_auto) so the mutual-match
flow is demonstrable end to end.
"""

from datetime import datetime, timezone

from app.services.mongo import connect_to_mongo

AUTO_RECIPROCATE_THRESHOLD = 85.0


def _col(client):
    col = client["dormmate"]["interests"]
    col.create_index([("from_user", 1), ("to_user", 1)], unique=True)
    return col


def record_interest(from_user, to_user, action, score=None):
    """Upsert a like/skip. Returns whether this created a mutual match."""
    client = connect_to_mongo()
    if not client:
        return {"success": False, "message": "MongoDB connection failed."}
    try:
        col = _col(client)
        now = datetime.now(timezone.utc)
        col.replace_one(
            {"from_user": from_user, "to_user": to_user},
            {
                "from_user": from_user,
                "to_user": to_user,
                "action": action,
                "score": score,
                "created_at": now,
            },
            upsert=True,
        )

        mutual = False
        if action == "like":
            reverse = col.find_one(
                {"from_user": to_user, "to_user": from_user, "action": "like"}
            )
            if reverse:
                mutual = True
            elif score is not None and score >= AUTO_RECIPROCATE_THRESHOLD:
                col.replace_one(
                    {"from_user": to_user, "to_user": from_user},
                    {
                        "from_user": to_user,
                        "to_user": from_user,
                        "action": "like",
                        "score": score,
                        "demo_auto": True,
                        "created_at": now,
                    },
                    upsert=True,
                )
                mutual = True

        return {"success": True, "mutual": mutual}
    finally:
        client.close()


def _entry(doc, mutual):
    return {
        "name": doc["to_user"],
        "score": doc.get("score"),
        "mutual": mutual,
        "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
    }


def get_sent(user):
    """Likes this user has sent, each flagged mutual/pending."""
    client = connect_to_mongo()
    if not client:
        return {"success": False, "message": "MongoDB connection failed."}
    try:
        col = _col(client)
        sent = list(col.find({"from_user": user, "action": "like"}))
        if not sent:
            return {"success": True, "sent": []}

        names = [d["to_user"] for d in sent]
        reciprocals = {
            d["from_user"]
            for d in col.find(
                {"from_user": {"$in": names}, "to_user": user, "action": "like"}
            )
        }
        return {
            "success": True,
            "sent": [_entry(d, d["to_user"] in reciprocals) for d in sent],
        }
    finally:
        client.close()


def get_mutual(user):
    """Pairs where both directions are likes."""
    result = get_sent(user)
    if not result["success"]:
        return result
    return {
        "success": True,
        "mutual": [e for e in result["sent"] if e["mutual"]],
    }
