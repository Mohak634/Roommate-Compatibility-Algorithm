"""One shared MongoDB client for the whole app.

Configure with MONGODB_URI, or MONGO_USER / MONGO_PASS / MONGO_CLUSTER.
MONGODB_URI=mongomock:// gives an in-memory database (tests, quick local runs).
If nothing is configured, get_db() returns None and the app runs read-only on
the bundled sample profiles.
"""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


def _uri() -> str | None:
    uri = os.getenv("MONGODB_URI")
    if uri:
        return uri
    user, password, cluster = os.getenv("MONGO_USER"), os.getenv("MONGO_PASS"), os.getenv("MONGO_CLUSTER")
    if not (user and password and cluster):
        return None
    app_name = os.getenv("MONGO_APP", "DormMate0")
    return f"mongodb+srv://{quote_plus(user)}:{quote_plus(password)}@{cluster}/?appName={quote_plus(app_name)}"


@lru_cache(maxsize=1)
def get_client():
    uri = _uri()
    if uri is None:
        return None
    if uri.startswith("mongomock://"):
        import mongomock

        return mongomock.MongoClient()
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi

    kwargs = {"server_api": ServerApi("1")} if uri.startswith("mongodb+srv://") else {}
    return MongoClient(uri, serverSelectionTimeoutMS=5000, **kwargs)


def get_db():
    client = get_client()
    if client is None:
        return None
    db = client[os.getenv("MONGO_DB", "dormmate")]
    _ensure_indexes(db)
    return db


_indexed: set[int] = set()


def _ensure_indexes(db) -> None:
    if id(db.client) in _indexed:
        return
    db.users.create_index("username", unique=True)
    db.users.create_index("email", unique=True)
    db.profiles.create_index("id", unique=True)
    db.interests.create_index([("from_user", 1), ("to_user", 1)], unique=True)
    _indexed.add(id(db.client))


def reset() -> None:
    """Forget the client (tests)."""
    get_client.cache_clear()
    _indexed.clear()
