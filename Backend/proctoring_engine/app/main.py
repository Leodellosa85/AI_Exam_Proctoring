from fastapi import FastAPI
from .api.rest import router as rest_router
from .api.websocket.v1.routers import router as ws_v1_router
from .api.websocket.v2.routers import router as ws_v2_router
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
app.include_router(ws_v1_router, prefix="/ws/v1")
app.include_router(ws_v2_router, prefix="/ws/v2")

@app.get("/health")
async def health():
    return {"status": "ok"}




