import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints.endpoints import router  # import the router you created

# Windows consoles default to cp1252; the codebase prints ✅/❌ in a few
# places. Force UTF-8 so a log line can never crash a request handler.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

app = FastAPI(title="DormMate API")

# Allow local dev frontends (Flutter web uses a dynamic port; Next.js uses
# 3000). Auth is stateless JSON (no cookies), so a wildcard origin is fine
# here — keep allow_credentials False to satisfy the browser's CORS rules.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount your router under the '/api' prefix (already set in the router)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
