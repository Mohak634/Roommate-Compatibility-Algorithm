from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.mongo import (
    signup_user,
    login_user)

from app.services.interests import (
    record_interest,
    get_sent,
    get_mutual,
)

from app.services.survey import get_survey_schema
from app.services.profiles import save_profile, get_profile

from app.core.main_script import (
    enablegpu,
    fetch_google_sheet,
    clean_data,
    preprocess_data,
    preprocess_testdata,
    encode_data,
    train_cluster,
    train_model,
    test_with_clusters,
    test_with_model,
    test_matching,
    display_top_matches,
    plot_radar_chart,
    compare_two_users
)

# Web-safe matching wrappers (no input()/plt.show(); return JSON).
from app.core.web_matching import (
    list_users,
    get_matches,
    compare_users_json,
)

# Create a FastAPI router with a prefix for grouping
router = APIRouter(prefix="/api", tags=["DormMate API"])

# Request models
class UserInput(BaseModel):
    name: str
    gender: Optional[str] = None  # for future use if needed


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username_or_email: str
    password: str

class CompareRequest(BaseModel):
    user1: str
    user2: str
    use_model: Optional[bool] = True

class MatchRequest(BaseModel):
    name: str
    use_model: Optional[bool] = True
    top_n: Optional[int] = 10

class InterestRequest(BaseModel):
    from_user: str
    to_user: str
    action: str  # "like" | "skip"
    score: Optional[float] = None

class ProfileSaveRequest(BaseModel):
    answers: dict

# ---------------------------------------------------------------------------
# Web-safe matching endpoints (consumed by the Next.js app)
# ---------------------------------------------------------------------------

@router.get("/users")
def api_list_users():
    """Return all candidate names so the UI can offer a picker."""
    return {"users": list_users()}

@router.post("/matches")
def api_get_matches(req: MatchRequest):
    """Ranked roommate matches for a given person, as JSON."""
    result = get_matches(req.name, use_model=req.use_model, top_n=req.top_n)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@router.post("/compare")
def api_compare_json(data: CompareRequest):
    """Compare two users, returning score + shared factors (no plot)."""
    result = compare_users_json(data.user1, data.user2, use_model=data.use_model)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@router.post("/interest")
def api_record_interest(data: InterestRequest):
    """Record a like/skip; returns whether it became a mutual match."""
    if data.action not in ("like", "skip"):
        raise HTTPException(status_code=422, detail="action must be 'like' or 'skip'")
    result = record_interest(data.from_user, data.to_user, data.action, data.score)
    if not result["success"]:
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@router.get("/interests/sent/{user}")
def api_interests_sent(user: str):
    """Likes the user has sent, each flagged mutual or pending."""
    result = get_sent(user)
    if not result["success"]:
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@router.get("/matches/mutual/{user}")
def api_mutual_matches(user: str):
    """Pairs where both people liked each other."""
    result = get_mutual(user)
    if not result["success"]:
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@router.get("/survey/schema")
def api_survey_schema():
    """Survey questions/options, straight from the encoder's vocabulary."""
    return get_survey_schema()

@router.get("/profile/{username}")
def api_get_profile(username: str):
    """Saved survey answers for a user (answers=null if none yet)."""
    result = get_profile(username)
    if not result["success"]:
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@router.post("/profile/{username}")
def api_save_profile(username: str, data: ProfileSaveRequest):
    """Validate + save survey answers; marks the user's profile complete."""
    result = save_profile(username, data.answers)
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["message"])
    return result

# Endpoints

@router.post("/enable_gpu")
def api_enable_gpu():
    enablegpu()
    return {"status": "GPU enabled"}

@router.post("/fetch_google_sheet")
def api_fetch_google_sheet():
    fetch_google_sheet()
    return {"status": "Google Sheet saved to RawData.csv"}

@router.post("/clean_data")
def api_clean_data():
    clean_data()
    return {"status": "Data Cleaned and saved to Cleaned.csv"}

@router.post("/encode_data")
def api_encode_data():
    encode_data()
    return {"status": "Data encoded"}

@router.post("/preprocess_data")
def api_preprocess_data():
    processed = preprocess_data()
    return {"status": "Data preprocessed"}

@router.post("/preprocess_datatest")
def api_preprocess_testdata():
    processed_test = preprocess_testdata()
    return {"status": "Test data preprocessed"}

@router.post("/train_cluster")
def api_train_cluster():
    train_cluster()
    return {"status": "Clustering model trained"}

@router.post("/train_model")
def api_train_model():
    train_model()
    return {"status": "Neural network model trained"}

@router.post("/test_with_clusters")
def api_test_with_clusters():
    results = test_with_clusters()
    return {"results": results}

@router.post("/test_with_models")
def api_test_with_models():
    results = test_with_model()
    return {"results": results}

@router.post("/test_matching")
def api_test_matching(user: UserInput):
    result = test_matching(user.name)
    return {"matches": result}

@router.post("/display_top_matches")
def api_display_top_matches(user: UserInput):
    top_matches = display_top_matches(user.name)
    return {"top_matches": top_matches}

@router.post("/plot_radar_chart")
def api_plot_radar_chart(user: UserInput):
    radar = plot_radar_chart(user.name)
    return {"radar_chart": radar}

@router.post("/Sign Up New User")
def api_signup(data: SignupRequest):
    result = signup_user(data.username, data.email, data.password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@router.post("/Login User")
def api_login(data: LoginRequest):
    result = login_user(data.username_or_email, data.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return result

@router.post("/Compare two users")
def api_compare(data: CompareRequest):
    try:
        compare_two_users(data.user1, data.user2, use_model=data.use_model)
        return {
            "success": True,
            "message": f"Comparison between {data.user1} and {data.user2} complete. Radar plot shown.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")