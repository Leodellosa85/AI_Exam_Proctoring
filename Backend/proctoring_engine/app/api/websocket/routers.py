from fastapi import APIRouter, WebSocket
# from .websocket import DirectionWebSocket
# from .websocketv2 import DirectionWebSocketV2
# from .websocketV3 import DirectionWebSocketV2
# from ...detection.face_detector_v2 import HF_MediaPipe_Detector
from .websocket import LivenessWebSocket

router = APIRouter()

# detector_model = HF_MediaPipe_Detector()

# ws_handler_v1 = DirectionWebSocket(detector=detector_model)
# ws_handler_v2 = DirectionWebSocketV2()

ws_handler = LivenessWebSocket()

# @router.websocket("/ws/{session_id}")
# async def websocket_endpoint_v1(websocket: WebSocket, session_id: str):
#     await ws_handler_v1.handle(websocket, session_id)

# @router.websocket("/ws2/{session_id}")
# async def websocket_endpoint_v2(websocket: WebSocket, session_id: str):
#     await ws_handler_v2.handle(websocket, session_id)

@router.websocket("/ws2/{session_id}")
async def websocket_endpoint_v2(websocket: WebSocket, session_id: str):
    await ws_handler.handle(websocket, session_id)
