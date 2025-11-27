from fastapi import APIRouter, WebSocket
from .websocket import DirectionWebSocketV2
# from some_model_loader import load_detector  # adjust to your path

router = APIRouter()

# detector_model = load_detector()  # your ML model
# ws_handler = DirectionWebSocketV2(detector=detector_model)

# @router.websocket("/ws-endpoint")
# async def websocket_endpoint(websocket: WebSocket):
#     await ws_handler.handle(websocket)
