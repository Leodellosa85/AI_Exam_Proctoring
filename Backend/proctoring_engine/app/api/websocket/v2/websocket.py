from fastapi import WebSocket

class DirectionWebSocketV2:
    def __init__(self, detector=None):
        self.detector = detector  # inject deep learning model
    
    async def handle(self, websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"status": "connected", "version": "v2"})

        while True:
            data = await websocket.receive_bytes()

            if not self.detector:
                await websocket.send_json({
                    "error": "MODEL_NOT_LOADED",
                    "message": "Direction detection model failed to load."
                })
                continue

            direction = self.detector.predict(data)

            if direction == -1:
                await websocket.send_json({
                    "error": "DETECTION_FAILED",
                    "message": "Unable to detect direction."
                })
                continue

            await websocket.send_json({
                "version": "v2",
                "direction": direction
            })
