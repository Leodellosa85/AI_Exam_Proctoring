import io
import time
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session, save_session_log
from ...core.logger import log_event
from ...core.counters import handle_multiperson, handle_lookaway
from ...config.settings import FACE_ABSENCE_TIMEOUT, LOOKAWAY_MIN_ANGLE

class DirectionWebSocket:
    """
    WebSocket handler using a hybrid HF + MediaPipe detector
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

                # Detect faces
                num_faces = self.detector.detect_faces(img) if self.detector else -1
                now = time.time()

                if num_faces == -1:
                    # Model failed
                    await websocket.send_json({
                        "error": "MODEL_NOT_LOADED",
                        "message": "Face detection model failed."
                    })
                    continue

                # Face absence
                if num_faces == 0:
                    if now - session["last_face_seen"] > FACE_ABSENCE_TIMEOUT:
                        session["terminated"] = True
                        log_event(session, "terminate_exam", {"reason": "face_absent"})
                        await websocket.send_json({"terminate": True})
                        save_session_log(session_id)
                        break
                else:
                    session["last_face_seen"] = now

                # Multi-person detection
                handle_multiperson(session, num_faces)

                # Head pose / direction detection
                yaw, pitch = self.detector.analyze_head_pose(img)
                handle_lookaway(session, yaw, pitch, LOOKAWAY_MIN_ANGLE)

                # Send results
                await websocket.send_json({
                    "num_faces": num_faces,
                    "yaw": yaw,
                    "pitch": pitch,
                    "flags": {
                        "looking_away": session.get("flag_lookaway", False),
                        "multiperson": session.get("flag_multi", False)
                    },
                    "terminate": session["terminated"]
                })

                if session["terminated"]:
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)
