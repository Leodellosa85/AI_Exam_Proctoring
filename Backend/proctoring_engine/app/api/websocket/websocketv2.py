from fastapi import WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session, save_session_log
from ...core.logger import log_event
from ...core.spoof_detector import SpoofDetector
import time
import base64
import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque  # <--- Required for smoothing

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "debug_facesv3")
os.makedirs(DEBUG_DIR, exist_ok=True)

spoof_detector = SpoofDetector()

class DirectionWebSocketV2:
    def __init__(self, detector=None):
        self.detector = detector

    def _decode_face(self, b64_img: str):
        try:
            if "," in b64_img:
                b64_img = b64_img.split(",", 1)[1]
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

        # Initialize session state and score buffer (Smoothing over last 5 frames)
        session["spoof_score"] = None
        session["flag_spoof"] = False
        if "score_buffer" not in session:
            session["score_buffer"] = deque(maxlen=5)

        log_event(session, "ws_connected", {})
        print("WebSocket V2 connected:", session_id)

        try:
            while True:
                data = await websocket.receive_json()

                if data.get("type") == "face_crop":
                    face_img = self._decode_face(data.get("image"))

                    if face_img is None:
                        print("❌ Face image decode failed")
                        continue

                    # ---- SAVE DEBUG IMAGE ----
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"face_{ts}.jpg"
                    cv2.imwrite(os.path.join(DEBUG_DIR, filename), face_img)

                    # 1. Get raw prediction
                    raw_score = spoof_detector.predict(face_img)
                    
                    if raw_score is not None:
                        # 2. Add to buffer and calculate stable average
                        session["score_buffer"].append(raw_score)
                        stable_score = sum(session["score_buffer"]) / len(session["score_buffer"])
                        
                        session["spoof_score"] = stable_score
                        
                        # 3. Decision Logic (Requires at least 3 frames for stability)
                        if len(session["score_buffer"]) >= 3:
                            # Using 0.65 threshold for the smoothed score
                            if stable_score > 0.65:
                                session["flag_spoof"] = True
                                log_event(session, "spoof_detected", {"score": stable_score})
                            else:
                                session["flag_spoof"] = False
                        
                        print(f"Raw: {raw_score:.4f} | Stable: {stable_score:.4f} | BuffLen: {len(session['score_buffer'])}")

                        # ✅ Send result back to frontend
                        await websocket.send_json({
                            "type": "spoof_result",
                            "liveness": "fake" if session["flag_spoof"] else "real",
                            "spoof_score": float(stable_score),
                            "flags": {
                                "spoof": session["flag_spoof"]
                            },
                            "terminate": session.get("terminated", False)
                        })

                elif data.get("type") == "heartbeat":
                    session["last_face_seen"] = time.time()

                if session.get("terminated"):
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)
