"""Accounts and signed login tokens.

Login returns a JWT (HS256, signed with JWT_SECRET). Protected routes read the
user from the token via the `current_user` dependency, never from the URL.
"""

from __future__ import annotations

import os
import secrets
import warnings
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo.errors import DuplicateKeyError

from app.services.db import get_db

TOKEN_DAYS = 7
_bearer = HTTPBearer(auto_error=False)
_dev_secret = None


def _secret() -> str:
    global _dev_secret
    secret = os.getenv("JWT_SECRET")
    if secret:
        return secret
    if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("PRODUCTION"):
        raise RuntimeError("JWT_SECRET must be set in production.")
    if _dev_secret is None:
        warnings.warn("JWT_SECRET is not set; using a random secret (tokens reset on restart).")
        _dev_secret = secrets.token_urlsafe(32)
    return _dev_secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": username, "iat": now, "exp": now + timedelta(days=TOKEN_DAYS)},
                      _secret(), algorithm="HS256")


def _public(user: dict) -> dict:
    return {"username": user["username"], "email": user["email"],
            "profile_complete": bool(user.get("profile_complete"))}


def _require_db():
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    return db


def signup(username: str, email: str, password: str) -> dict:
    db = _require_db()
    username, email = username.strip(), email.strip().lower()
    if len(username) < 3 or len(password) < 8 or "@" not in email:
        raise HTTPException(status_code=422,
                            detail="Username needs 3+ characters, password 8+, and a valid email.")
    if username.startswith("sample-"):
        raise HTTPException(status_code=422, detail="That username is reserved.")
    user = {"username": username, "email": email, "password_hash": hash_password(password),
            "created_at": datetime.now(timezone.utc), "profile_complete": False}
    try:
        db.users.insert_one(user)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Username or email already exists.")
    return {"token": create_token(username), "user": _public(user)}


def login(username_or_email: str, password: str) -> dict:
    db = _require_db()
    key = username_or_email.strip()
    user = db.users.find_one({"$or": [{"username": key}, {"email": key.lower()}]})
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    return {"token": create_token(user["username"]), "user": _public(user)}


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Log in first.")
    try:
        username = jwt.decode(creds.credentials, _secret(), algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Session expired. Log in again.")
    user = _require_db().users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="Account not found.")
    return _public(user)
