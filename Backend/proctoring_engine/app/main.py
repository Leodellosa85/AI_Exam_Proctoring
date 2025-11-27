from fastapi import FastAPI
from .api.rest import router as rest_router
from .api.websocket.routers import router as ws_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Proctoring Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000", "http://127.0.0.1:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# include_router() adds route groups (routers) to the main app.

# rest_router likely contains HTTP REST endpoints
app.include_router(rest_router)
# ws_router likely contains WebSocket endpoints for real-time communication (like proctoring video/audio streams or notifications).
app.include_router(ws_router)

@app.get("/health")
async def health():
    return {"status": "ok"}




