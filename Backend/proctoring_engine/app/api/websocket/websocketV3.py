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
import json

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "debug_facesv3")
os.makedirs(DEBUG_DIR, exist_ok=True)

spoof_detector = SpoofDetector()   # load once (important)

class DirectionWebSocketV2:

    def __init__(self, detector=None):
        self.detector = detector

    def _decode_face(self, b64_img: str):
        try:
            if "," in b64_img:
                b64_img = b64_img.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_img)
            return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print("Base64 decode error:", e)
            return None
        
    def check_image_quality(self, face_img):
        """Returns (is_valid, reason)"""
        # Convert to grayscale for calculations
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # --- 1. Brightness Check ---
        # Calculate average pixel intensity (0-255)
        avg_brightness = np.mean(gray)
        if avg_brightness > 230: return False, "too_bright" # Screen washout
        if avg_brightness < 40:  return False, "too_dark"   # Poor lighting
        
        # --- 2. Resolution/Sharpness Check (Laplacian Variance) ---
        # Real skin has a specific texture depth that blur destroys
        focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if focus_score < 100: return False, "blurry" # Adjust 100 based on your camera
        
        return True, "ok"


    async def handle(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        session = get_session(session_id)

        if not session:
            session_id, session = create_session()

        # Initialize safely
        session["spoof_score"] = None
        session["flag_spoof"] = False
        session.setdefault("spoof_history", [])

         # Track the last JSON header to know what the next binary chunk belongs to
        last_msg_type = None 

        print(f"WebSocket V2 (Binary Mode) connected: {session_id}")

        log_event(session, "ws_connected", {})

        try:
            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    print("Client disconnected cleanly")
                    break

                # --- HANDLE TEXT DATA (JSON Headers / Heartbeats) ---
                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "face_crop":
                        last_msg_type = "face_crop" # Prep for next binary chunk
                    
                    elif msg_type == "heartbeat":
                        session["last_face_seen"] = time.time()
                        last_msg_type = "heartbeat"

                # --- HANDLE BINARY DATA (The actual Face Image) ---
                elif "bytes" in message:
                    if last_msg_type == "face_crop":
                        binary_data = message["bytes"]
                        
                        # 1. Faster Decode: Direct from bytes to OpenCV
                        nparr = np.frombuffer(binary_data, np.uint8)
                        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if face_img is None or face_img.size == 0:
                            continue

                        is_valid, reason = self.check_image_quality(face_img)
    
                        if not is_valid:
                            # Skip inference if quality is bad, but tell the frontend why
                            await websocket.send_json({
                                "type": "spoof_result",
                                "liveness": "unstable",
                                "reason": reason,
                                "spoof_score": session.get("spoof_score", 0)
                            })
                            continue # Skip to next frame

                        # 2. Process through FaceBagNet Decision Engine
                        # (Same logic as your previous version, but using stable binary data)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"face_{ts}.jpg"
                        cv2.imwrite(os.path.join(DEBUG_DIR, filename), face_img)

                        spoof_score = spoof_detector.predict(face_img)
                        
                        if spoof_score is not None:
                            session["spoof_history"].append(spoof_score)
                            if len(session["spoof_history"]) > 5:
                                session["spoof_history"].pop(0)
                            
                            avg_score = sum(session["spoof_history"]) / len(session["spoof_history"])

                            # --- Final Decision Logic ---
                            if avg_score >= 0.90:
                                liveness = "real"
                            elif avg_score >= 0.85:
                                liveness = "suspicious"
                            else:
                                liveness = "fake"

                            session["spoof_score"] = avg_score
                            session["flag_spoof"] = (liveness == "fake")

                            await websocket.send_json({
                                "type": "spoof_result",
                                "liveness": liveness,
                                "spoof_score": float(avg_score),
                                "flags": {"spoof": session["flag_spoof"]},
                                "terminate": False
                            })
                        
                        last_msg_type = None # Reset after processing

                if session.get("terminated"):
                    break

        except WebSocketDisconnect:
            log_event(session, "ws_disconnected", {})
            save_session_log(session_id)

        except RuntimeError as e:
            print("Runtime WebSocket error:", e)

        except Exception as e:
            print("Unexpected WS error:", e)

