import io
import time
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session, save_session_log
from ...core.logger import log_event
from ...config.settings import FACE_ABSENCE_TIMEOUT


class DirectionWebSocket:
    """
    WebSocket handler for:
    ✔ Face detection
    ✔ Liveness check (real vs fake)
    ✔ Normalized bounding boxes for frontend overlay
    """

    def __init__(self, detector=None):
        self.detector = detector

    async def handle(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        session = get_session(session_id)

        if not session:
            session_id, session = create_session()

        log_event(session, "ws_connected", {})

        try:
            while True:
                # Receive binary image frame
                frame_bytes = await websocket.receive_bytes()
                img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                img_width, img_height = img.size

                # ----------------------------------------------------
                # 1. FACE DETECTION
                # ----------------------------------------------------
                if self.detector:
                    num_faces, boxes = self.detector.detect_faces(img)
                else:
                    num_faces, boxes = 0, []

                now = time.time()

                if num_faces == -1:
                    await websocket.send_json({
                        "error": "MODEL_NOT_LOADED",
                        "message": "Face detection model failed."
                    })
                    continue

                # No detected faces
                if num_faces == 0:
                    if now - session["last_face_seen"] > FACE_ABSENCE_TIMEOUT:
                        session["terminated"] = True
                        log_event(session, "terminate_exam", {"reason": "face_absent"})
                        await websocket.send_json({"terminate": True})
                        save_session_log(session_id)
                        break
                else:
                    session["last_face_seen"] = now

                # ----------------------------------------------------
                # 2. LIVENESS CHECK (Real vs Fake face)
                # ----------------------------------------------------
                liveness_status = "unknown"

                if num_faces == 1:
                    # Optional: crop face before liveness
                    # For now – send full image to model
                    liveness_status = self.detector.check_liveness(img)
                    session["flag_spoof"] = (liveness_status != "real")
                    if session["flag_spoof"]:
                        log_event(session, "spoof_detected", {"status": liveness_status})
                else:
                    session["flag_spoof"] = False

                # ----------------------------------------------------
                # 3. Normalize boxes for frontend overlay
                # ----------------------------------------------------
                boxes_normalized = []
                for box in boxes:
                    boxes_normalized.append({
                        "x": box["x"] / img_width,
                        "y": box["y"] / img_height,
                        "width": box["width"] / img_width,
                        "height": box["height"] / img_height
                    })

                # ----------------------------------------------------
                # 4. Send response to client
                # ----------------------------------------------------
                await websocket.send_json({
                    "num_faces": num_faces,
                    "boxes": boxes_normalized,
                    "liveness": liveness_status,
                    "flags": {
                        "spoof": session.get("flag_spoof", False),
                    },
                    "terminate": session["terminated"]
                })

                if session["terminated"]:
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)
