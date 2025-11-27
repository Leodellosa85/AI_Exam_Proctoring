# app/api/websocket/v1/routers.py
from fastapi import APIRouter, WebSocket
from .websocket import DirectionWebSocketV1

router = APIRouter()
ws_handler = DirectionWebSocketV1()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_handler.handle(websocket, session_id)
