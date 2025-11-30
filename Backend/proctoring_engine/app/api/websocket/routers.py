from fastapi import APIRouter, WebSocket
from .websocket import DirectionWebSocket
from ...detection.face_detector_v2 import HF_MediaPipe_Detector

router = APIRouter()

detector_model = HF_MediaPipe_Detector()
ws_handler = DirectionWebSocket(detector=detector_model)

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_handler.handle(websocket, session_id)
