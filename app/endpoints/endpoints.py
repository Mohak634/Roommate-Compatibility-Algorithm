"""DormMate API routes. Everything except signup, login, the survey schema and
health needs `Authorization: Bearer <token>` from login or signup."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.services import auth, interests, matches, profile_store, profiles
from app.services.db import get_db
from app.services.survey import get_survey_schema

router = APIRouter(prefix="/api", tags=["DormMate API"])


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class AnswersRequest(BaseModel):
    answers: dict


class InterestRequest(BaseModel):
    profile_id: str
    action: str  # "like" | "skip"


# --- Auth -------------------------------------------------------------------

@router.post("/auth/signup", status_code=201)
def signup(data: SignupRequest):
    return auth.signup(data.username, data.email, data.password)


@router.post("/auth/login")
def login(data: LoginRequest):
    return auth.login(data.username_or_email, data.password)


@router.get("/me")
def me(user: dict = Depends(auth.current_user)):
    return user


# --- Survey and profile -----------------------------------------------------

@router.get("/survey/schema")
def survey_schema():
    return get_survey_schema()


@router.get("/profile")
def get_my_profile(user: dict = Depends(auth.current_user)):
    p = profile_store.get_profile(user["username"])
    return {"profile": matches.card(p) if p else None, "answers": p["answers"] if p else None}


@router.put("/profile")
def save_my_profile(data: AnswersRequest, user: dict = Depends(auth.current_user)):
    p = profiles.save(user["username"], data.answers)
    return {"profile": matches.card(p), "answers": p["answers"]}


# --- Matches ----------------------------------------------------------------

@router.get("/matches")
def my_matches(top_n: int = Query(10, ge=1, le=50), user: dict = Depends(auth.current_user)):
    return matches.list_matches(user["username"], top_n)


@router.get("/matches/mutual")
def my_mutual(user: dict = Depends(auth.current_user)):
    return {"mutual": interests.mutual(user["username"])}


@router.get("/matches/{profile_id}")
def match_detail(profile_id: str, user: dict = Depends(auth.current_user)):
    mutual = interests.is_mutual(user["username"], profile_id)
    other_email = None
    if mutual:
        acct = get_db().users.find_one({"username": profile_id})
        other_email = acct["email"] if acct else None
    return matches.match_detail(user["username"], profile_id, mutual, other_email)


# --- Interests --------------------------------------------------------------

@router.post("/interests")
def express_interest(data: InterestRequest, user: dict = Depends(auth.current_user)):
    return interests.record(user["username"], data.profile_id, data.action)


@router.get("/interests/sent")
def sent_interests(user: dict = Depends(auth.current_user)):
    return {"sent": interests.sent(user["username"])}
