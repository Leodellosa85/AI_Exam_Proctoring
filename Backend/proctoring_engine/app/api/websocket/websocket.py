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
                # Receive binary frame
                frame_bytes = await websocket.receive_bytes()
                img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

                # ----------------------------------------------------
                # 1. FACE DETECTION
                # ----------------------------------------------------
                num_faces = self.detector.detect_faces(img) if self.detector else -1
                now = time.time()

                if num_faces == -1:
                    await websocket.send_json({
                        "error": "MODEL_NOT_LOADED",
                        "message": "Face detection model failed."
                    })
                    continue

                # Face not found
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
                # 2. LIVENESS CHECK (Real vs Fake)
                # ----------------------------------------------------
                liveness_status = "unknown"

                if num_faces == 1:
                    # Full image or crop — based on your detector implementation
                    liveness_status = self.detector.check_liveness(img)

                    # Mark spoof flag
                    session["flag_spoof"] = (liveness_status != "real")

                    if session["flag_spoof"]:
                        log_event(session, "spoof_detected", {"status": liveness_status})

                else:
                    # If more than one face, you choose behavior:
                    # For now: "unknown", not terminating exam
                    liveness_status = "unknown"

                # ----------------------------------------------------
                # 3. SEND RESPONSE TO CLIENT
                # ----------------------------------------------------
                await websocket.send_json({
                    "num_faces": num_faces,
                    "liveness": liveness_status,
                    "flags": {
                        "spoof": session.get("flag_spoof", False)
                    },
                    "terminate": session["terminated"]
                })

                if session["terminated"]:
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)
