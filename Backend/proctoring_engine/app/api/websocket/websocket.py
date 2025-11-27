from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session, save_session_log
from ...core.logger import log_event
from ...core.counters import handle_multiperson, handle_lookaway
from ...utils.image import base64_to_bytes, pil_from_bytes
from ...detection.mediapipe_detector import MediapipeDetector
from ...config.settings import FACE_ABSENCE_TIMEOUT, LOOKAWAY_MIN_ANGLE

import time
from PIL import Image
import io

router = APIRouter()
detector = MediapipeDetector()

class DirectionWebSocketV1:
    def __init__(self):
        self.detector = MediapipeDetector()

    async def handle(self, websocket: WebSocket, session_id: str):
        """Accept the WebSocket Connection"""
        await websocket.accept()

        session = get_session(session_id)
        if not session:
            session_id, session = create_session()

        log_event(session, "ws_connected", {})

        try:
            while True:
                # ---- RECEIVE RAW BINARY BYTES ----
                frame_bytes: bytes = await websocket.receive_bytes()

                # Convert binary → PIL Image
                img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

                num_faces = self.detector.detect_faces(img)
                now = time.time()

                # ---- Face Absence Detection ----
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
                multi = handle_multiperson(session, num_faces)

                # ---- Head Pose / Lookaway ----
                yaw, pitch = self.detector.analyze_head_pose(img)
                handle_lookaway(session, yaw, pitch, LOOKAWAY_MIN_ANGLE)

                # ---- SEND BACK RESULTS ----
                await websocket.send_json({
                    "num_faces": num_faces,
                    "yaw": yaw,
                    "pitch": pitch,
                    "flags": {
                        "looking_away": session["flag_lookaway"],
                        "multiperson": session["flag_multi"]
                    },
                    "terminate": session["terminated"]
                })

                if session["terminated"]:
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)


@router.websocket("/wsb64/{session_id}")
async def ws_stream(ws: WebSocket, session_id: str):
    """Accept the WebSocket Connection"""
    await ws.accept()

    session = get_session(session_id)
    if not session:
        session_id, session = create_session()

    log_event(session, "ws_connected", {})

    """Main Real-Time Loop
    WebSocket stays open, so the server continuously:
    Receives image frame
    Processes it
    Sends results back instantly"""
    try:
        while True:
            # Browser sends Base64 string
            # Server converts it:
            # Base64 → Bytes → PIL Image (RGB)
            b64 = await ws.receive_text()
            img = pil_from_bytes(base64_to_bytes(b64))

            num_faces = detector.detect_faces(img)
            now = time.time()

            # face absence
            if num_faces == 0:
                if now - session["last_face_seen"] > FACE_ABSENCE_TIMEOUT:
                    session["terminated"] = True
                    log_event(session, "terminate_exam", {"reason": "face_absent"})
                    await ws.send_json({"terminate": True})
                    save_session_log(session_id)
                    break
            else:
                session["last_face_seen"] = now

            multi = handle_multiperson(session, num_faces)

            yaw, pitch = detector.analyze_head_pose(img)
            handle_lookaway(session, yaw, pitch, LOOKAWAY_MIN_ANGLE)

            await ws.send_json({
                "num_faces": num_faces,
                "yaw": yaw,
                "pitch": pitch,
                "flags": {
                    "looking_away": session["flag_lookaway"],
                    "multiperson": session["flag_multi"]
                },
                "terminate": session["terminated"]
            })

            if session["terminated"]:
                break

    except WebSocketDisconnect:
        log_event(session, "ws_disconnected", {})
        save_session_log(session_id)
