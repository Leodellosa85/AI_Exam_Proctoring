from fastapi import WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session, save_session_log
from ...core.logger import log_event
from ...core.spoof_detectorV2 import SpoofDetector
import time
import base64
import cv2
import numpy as np

import os
from datetime import datetime

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "debug_facesv3")
os.makedirs(DEBUG_DIR, exist_ok=True)

spoof_detector = SpoofDetector()   # load once (important)

class DirectionWebSocketV2:

    def __init__(self, detector=None):
        self.detector = detector

    def _decode_face(self, b64_img: str):
        try:
            if "," in b64_img:
                b64_img = b64_img.split(",", 1)[1]  # remove data:image/...;base64,

            img_bytes = base64.b64decode(b64_img)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            return img
        except Exception as e:
            print("Face decode error:", e)
            return None


    async def handle(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        session = get_session(session_id)

        if not session:
            session_id, session = create_session()

        # Initialize safely
        session["spoof_score"] = None
        session["flag_spoof"] = False

        log_event(session, "ws_connected", {})
        print("WebSocket V2 connected:", session_id)

        try:
            while True:
                data = await websocket.receive_json()

                # --------------------------------------------------
                # FACE CROP → SPOOF DETECTION
                # --------------------------------------------------
                if data.get("type") == "face_crop":

                    session.setdefault("spoof_frame_count", 0)
                    session.setdefault("spoof_history", [])

                    session["spoof_frame_count"] += 1
                    # if session["spoof_frame_count"] % 6 != 0:
                    #     continue

                    face_img = self._decode_face(data.get("image"))

                    if face_img is None or face_img.size == 0:
                        continue

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"face_{ts}.jpg"
                    cv2.imwrite(os.path.join(DEBUG_DIR, filename), face_img)
                    print("Saved:", filename)

                    spoof_score = spoof_detector.predict(face_img)
                    print(f"Spoof score: {spoof_score}")

                    if spoof_score is not None:

                        session["spoof_history"].append(spoof_score)
                        if len(session["spoof_history"]) > 5:
                            session["spoof_history"].pop(0)

                        avg_score = sum(session["spoof_history"]) / len(session["spoof_history"])

                        # if avg_score > 0.70:
                        #     liveness = "real"
                        # else:
                        #     liveness = "fake"

                        if avg_score > 0.90:
                            liveness = "real"      # Your face is consistently 0.92+
                        elif avg_score > 0.85:
                            liveness = "suspicious" # Transition zone
                        else:
                            liveness = "fake"

                        session["spoof_score"] = avg_score
                        session["flag_spoof"] = (liveness == "fake")

                        await websocket.send_json({
                            "type": "spoof_result",
                            "liveness": liveness,
                            "spoof_score": avg_score,
                            "flags": {"spoof": session["flag_spoof"]},
                            "terminate": False
                        })

                # --------------------------------------------------
                # HEARTBEAT
                # --------------------------------------------------
                elif data.get("type") == "heartbeat":
                    session["last_face_seen"] = time.time()

                if session.get("terminated"):
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)
