from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.mongo import (
    signup_user,
    login_user) 

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