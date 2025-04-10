from fastapi import FastAPI
from app.endpoints.endpoints import router  # import the router you created

app = FastAPI()

# Mount your router under the '/api' prefix (already set in the router)
app.include_router(router)